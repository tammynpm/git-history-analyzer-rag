# Git Commit History Analyzer 
A RAG system that indexes your git commit history and lets you search for similar past bug fixes using natural language. This project makes metadata first-class, i.e. you can search by uathor, date range, files changed, fix patterns, and semantic similarity simultaneously. 
Optimized for local hardware (20GB RAM, CPU inference via Ollama). 

1. Parsing 
Raw `git log` output is unstructured text. We parse each commit into a structured object:
```
GitCommit {
    hash: str 
    short_hash: str
    author: str
    author_email: str
    date: datetime
    message: str
    subject: str
    files_changed: list[str]
    insertions: int
    deletions: int
    is_bug_fix: bool
    fix_category: str
    components: list[str]
}
```

Not every commit is a bug fix. We classify commits using signal words in the subject/body:

|Words | Classification |
|------| ------|
|fix,bugfix,hotfix| bug_fix|
|resolve, closes #, fixes #| bug_fix|
|patch, workaround| bug_fix|
|crash, error, exception| bug_fix|
|feat, feature, add| feature|
|refactor, restructure| refactor|
|docs, readme, comment| documentation|
|test, spec, coverage| test|
|chore, bump, update dep| maintenance|


This classification becomes searchable metadata, so you can filter to only bug fixes before doing semantic search. 

2. Chunking 
A commit is already a natural semantic unit so the approach here is that one chunk = one commit. 

3. Metadata extraction 
We store structured metadata alongside each embeddings. 
```
metadata = {
    "hash": "123456789",
    "author": "tammy",
    "date": "2026-01-01", #ISO format string
    "timestamp": 1710460800, #Unix timestamp
    "is_bug_fix": True,
    "fix_category": "bug_fix",
    "components": "auth,session", #comma-separated for filtering 
    "files_changed": "src/auth/session.py,src/auth/middleware.py",
    "insertions": 42,
    "deletions": 18,
    "subject": "fix session timeout on slow connections"

}
```

4. Retrieval
We use a two-stage hybrid retrieval approach 
**Stage 1: Metadata pre-filtering**
**Stage 2: Semantic similarity**
Within the filtered results, rank by cosine similarity between the query embedding and chunk embeddings. This finds conceptually similar fixes even with different wording. 

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