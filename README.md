# Git Commit History Analyzer 
A RAG system that indexes your git commit history and lets you search for similar past bug fixes using natural language. This project makes metadata first-class, i.e. you can search by uathor, date range, files changed, fix patterns, and semantic similarity simultaneously. 
Optimized for local hardware (20GB RAM, CPU inference via Ollama). 

1. Parsing 
Raw `git log` output is unstructured text. We parse each commit into a structured object. 

Not every commit is a bug fix. We classify commits using signal words in the subject/body. 

This classification becomes searchable metadata, so you can filter to only bug fixes before doing semantic search. 

2. Chunking 
A commit is already a natural semantic unit so the approach here is that one chunk = one commit. 

3. Metadata extraction 
We store structured metadata alongside each embeddings. 

4. Retrieval
We use a two-stage hybrid retrieval approach 
**Stage 1: Metadata pre-filtering**
**Stage 2: Semantic similarity**
Within the filtered results, rank by cosine similarity between the query embedding and chunk embeddings. This finds conceptually similar fixes even with different wording. 

--- 
## Setup

Prerequisites:
`pip install sentence-transformers chromadb gitpython python-dateutil`

Install Ollama and pull the appropriate model 
```
ollama pull phi3:mini
ollama serve
```

Index a repostiory
```
python ingest.py <path-to-repo>
```

Search for similar fixes
```
python query.py "session timeout on slow connections"
```