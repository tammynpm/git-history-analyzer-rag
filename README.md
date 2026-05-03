# Git Commit History Analyzer 
A RAG system that indexes your git commit history and lets you search for similar past bug fixes using natural language. `git log --grep` only matches exact keywords and finds semantically simliar commits even when the wording is different. This project makes metadata first-class, i.e. you can search by author, date range, files changed, fix patterns, and semantic similarity simultaneously. 
Optimized for local hardware (20GB RAM, CPU inference via Ollama). 

## How It Works
1. Parsing 
Using GitPython to extract structured data from each commit: 
```
{
    "hash": "5a66a92f",
    "author": "tammynpm",
    "email": "112338317+tammynpm@users.noreply.github.com",
    "date": "2026-03-20",
    "message": "updated intro",
    "files_changed": ["README.md"],
    "insertions": 5,
    "deletions": 2
}
```
Files stats are included because commit messages are often vague. 

2. Chunking 
A commit is already a natural semantic unit so the approach here is that one chunk = one commit. 

3. Embedding & Storage
Each chunk is converted into a 384-dimensional vector using `sentence-transformers` (all-MiniLM-L6-v2) and stored in ChromaDB with basic metadata (author, date, message) 

4. Retrieval
When you ask a question, it gets embedded into a vector, ChromaDB finds the most similar commits by cosine similarity, and those commits are sent to Ollama (phi3:mini) as context to generate a human-readable answer.

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

### TODO
- [ ] classify commits into categories
- [ ] store full structured metadata
- [ ] accept repo path and query from command line instead of hardcoding 
