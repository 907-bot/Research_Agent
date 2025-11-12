# -*- coding: utf-8 -*-
"""
Research Agent - Web Search and Summarization Tool
Deployed on Hugging Face Spaces with Gradio
"""

import re
import urllib.parse
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import gradio as gr

# Configuration
SEARCH_RESULTS = 6
PASSAGES_PER_PAGE = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_PASSAGES = 5
SUMMARY_SENTENCES = 3
TIMEOUT = 8


def unwrap_ddg(url):
    """If DuckDuckGo returns a redirect wrapper, extract the real URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        if "duckduckgo.com" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg")
            if uddg:
                return urllib.parse.unquote(uddg[0])
    except Exception:
        pass
    return url


def search_web(query, max_results=SEARCH_RESULTS):
    """Search the web and return a list of URLs."""
    urls = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url")
                if not url:
                    continue
                url = unwrap_ddg(url)
                urls.append(url)
    except Exception as e:
        print(f"Search error: {e}")
    return urls


def fetch_text(url, timeout=TIMEOUT):
    """Fetch and clean text content from a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        if r.status_code != 200:
            return ""
        ct = r.headers.get("content-type", "")
        if "html" not in ct.lower():
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove unnecessary tags
        for tag in soup(["script", "style", "noscript", "header", "footer", 
                        "svg", "iframe", "nav", "aside"]):
            tag.extract()

        # Get paragraph text
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])

        if text.strip():
            return re.sub(r"\s+", " ", text).strip()

        # Fallback to meta description
        meta = soup.find("meta", attrs={"name": "description"}) or \
               soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            return meta["content"].strip()

        if soup.title and soup.title.string:
            return soup.title.string.strip()

    except Exception as e:
        print(f"Fetch error for {url}: {e}")
    return ""


def chunk_passages(text, max_words=120):
    """Split long text into smaller passages."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words
    return chunks


def split_sentences(text):
    """A simple sentence splitter."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


class ShortResearchAgent:
    def __init__(self, embed_model=EMBEDDING_MODEL):
        print(f"Loading embedder: {embed_model}...")
        self.embedder = SentenceTransformer(embed_model)

    def run(self, query, progress=gr.Progress()):
        """Run the research agent pipeline."""
        start = time.time()

        # Step 1: Search
        progress(0.1, desc="🔍 Searching the web...")
        urls = search_web(query)

        if not urls:
            elapsed = time.time() - start
            return {
                "query": query,
                "passages": [],
                "summary": "⚠️ No search results found. Please try a different query.",
                "time": elapsed,
                "num_urls": 0
            }

        # Step 2: Fetch & Chunk
        progress(0.3, desc=f"📥 Fetching content from {len(urls)} URLs...")
        docs = []
        for u in urls:
            txt = fetch_text(u)
            if not txt:
                continue
            chunks = chunk_passages(txt, max_words=120)
            for c in chunks[:PASSAGES_PER_PAGE]:
                docs.append({"url": u, "passage": c})

        if not docs:
            elapsed = time.time() - start
            return {
                "query": query,
                "passages": [],
                "summary": "⚠️ No content could be extracted from the search results.",
                "time": elapsed,
                "num_urls": len(urls)
            }

        # Step 3: Embed
        progress(0.5, desc="🧠 Analyzing content with AI...")
        texts = [d["passage"] for d in docs]
        emb_texts = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]

        # Step 4: Rank
        progress(0.7, desc="📊 Ranking relevant passages...")
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        sims = [cosine(e, q_emb) for e in emb_texts]
        top_idx = np.argsort(sims)[::-1][:TOP_PASSAGES]
        top_passages = [
            {
                "url": docs[i]["url"],
                "passage": docs[i]["passage"],
                "score": float(sims[i])
            }
            for i in top_idx
        ]

        # Step 5: Summarize
        progress(0.9, desc="✍️ Generating summary...")
        if not top_passages:
            summary = "⚠️ No relevant passages found for summarization."
        else:
            sentences = []
            for tp in top_passages:
                for s in split_sentences(tp["passage"]):
                    sentences.append({"sent": s, "url": tp["url"]})

            if not sentences:
                summary = "⚠️ No sentences found in relevant passages."
            else:
                sent_texts = [s["sent"] for s in sentences]
                sent_embs = self.embedder.encode(sent_texts, convert_to_numpy=True, 
                                                show_progress_bar=False)
                sent_sims = [cosine(e, q_emb) for e in sent_embs]
                top_sent_idx = np.argsort(sent_sims)[::-1][:SUMMARY_SENTENCES]
                chosen = [sentences[idx] for idx in top_sent_idx]

                # De-duplicate and format
                seen = set()
                lines = []
                for s in chosen:
                    key = s["sent"].lower()[:80]
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(f"{s['sent']} [(Source)]({s['url']})")

                summary = "\n\n".join(lines)

        elapsed = time.time() - start
        progress(1.0, desc="✅ Complete!")

        return {
            "query": query,
            "passages": top_passages,
            "summary": summary,
            "time": elapsed,
            "num_urls": len(urls)
        }


# Initialize the agent globally
print("Initializing Research Agent...")
agent = ShortResearchAgent()


def research_interface(query):
    """Gradio interface function."""
    if not query or len(query.strip()) < 3:
        return "❌ Please enter a valid query (at least 3 characters).", ""

    try:
        result = agent.run(query.strip())

        # Format summary
        summary_md = f"""# 📝 Research Summary

**Query:** {result['query']}

**Time taken:** {result['time']:.2f} seconds  
**URLs searched:** {result['num_urls']}

---

## Summary

{result['summary']}
"""

        # Format detailed passages
        passages_md = "# 🔍 Top Relevant Passages\n\n"
        if result['passages']:
            for i, p in enumerate(result['passages'], 1):
                passages_md += f"""### Passage {i} (Relevance: {p['score']:.2%})

**Source:** [{p['url']}]({p['url']})

{p['passage']}

---

"""
        else:
            passages_md += "No passages found."

        return summary_md, passages_md

    except Exception as e:
        error_msg = f"❌ **Error:** {str(e)}\n\nPlease try again with a different query."
        return error_msg, ""


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="AI Research Agent") as demo:
    gr.Markdown("""
    # 🤖 AI Research Agent

    ### Intelligent Web Search & Summarization Tool

    This tool searches the web, analyzes multiple sources, and provides you with:
    - **AI-generated summary** of the most relevant information
    - **Top passages** ranked by relevance with sources
    - **Fast results** powered by semantic search

    Simply enter your question below and let the AI do the research for you!
    """)

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="🔍 Enter your research query",
                placeholder="e.g., What causes urban heat islands and how can cities reduce them?",
                lines=2
            )
        with gr.Column(scale=1):
            search_btn = gr.Button("🚀 Research", variant="primary", size="lg")

    gr.Markdown("### 💡 Example Queries")
    with gr.Row():
        example_btns = [
            gr.Button("🌡️ Urban heat islands", size="sm"),
            gr.Button("🤖 Latest AI developments", size="sm"),
            gr.Button("🌱 Sustainable energy solutions", size="sm"),
            gr.Button("🧬 CRISPR gene editing", size="sm")
        ]

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            summary_output = gr.Markdown(label="Summary")

    with gr.Accordion("📚 Detailed Passages", open=False):
        passages_output = gr.Markdown(label="Top Passages")

    # Event handlers
    search_btn.click(
        fn=research_interface,
        inputs=[query_input],
        outputs=[summary_output, passages_output]
    )

    query_input.submit(
        fn=research_interface,
        inputs=[query_input],
        outputs=[summary_output, passages_output]
    )

    # Example button handlers
    example_queries = [
        "What causes urban heat islands and how can cities reduce them?",
        "What are the latest developments in artificial intelligence?",
        "What are the most promising sustainable energy solutions?",
        "How does CRISPR gene editing work and what are its applications?"
    ]

    for btn, query in zip(example_btns, example_queries):
        btn.click(
            fn=lambda q=query: q,
            outputs=[query_input]
        )

    gr.Markdown("""
    ---
    ### 📌 Tips
    - Be specific with your queries for better results
    - The tool analyzes 6 web sources by default
    - Results typically take 10-30 seconds depending on query complexity

    **Built with:** DuckDuckGo Search, Sentence Transformers, Gradio
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
