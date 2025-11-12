---
title: AI Research Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# AI Research Agent 🤖

An intelligent web search and summarization tool powered by AI.

## Features

- 🔍 Searches multiple web sources automatically
- 🧠 Uses semantic AI to understand and rank content
- 📝 Generates concise summaries from top sources
- ⚡ Fast results with progress tracking
- 🎨 Beautiful, user-friendly interface

## How it works

1. Enter your research question
2. The agent searches the web using DuckDuckGo
3. Content is fetched and analyzed using sentence transformers
4. Most relevant passages are ranked by semantic similarity
5. AI generates a summary from the best sources

## Technologies

- **Search**: DuckDuckGo Search API
- **AI Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Interface**: Gradio
- **Web Scraping**: BeautifulSoup4

## Usage

Simply type your question and click "Research" - the AI will do the rest!
