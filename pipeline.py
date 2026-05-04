import chromadb
import requests
from sentence_transformers import SentenceTransformer
from git import Repo

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="git_commits",
    metadata={"hnsw:space":"cosine"}
)

repo = Repo("/home/debian/rag/django")

chunks = []
ids = []
metadatas= []

BUG_FIX_SIGNALS= ["fix", "bug_fix", "bug fix", "hotfit", "bugfix",
                  "resolve", "closes #", "fixes #", 
                  "patch", "workaround",
                  "crash", "error", "exception"
                  ]

CATEGORY_SIGNALS = {
    "bug_fix": ["fix", "bug", "hotfix", "patch", "crash", "error", "solve"],
    "feature": ["feat", "feature", "add", "implement", "new"],
    "refactor": ["refactor", "restructure", "cleanup", "simplify"],
    "test": ["test", "spec", "coverage"],
    "documentation": ["doc", "readme", "content"],
}

def classify_commit(message: str):
    text = message.lower()
    is_bug_fix = any(signal in text for signal in BUG_FIX_SIGNALS)
    category="other"
    best_score=0
    for cat, signals in CATEGORY_SIGNALS.items():
        score=sum(1 for s in signals if signals in text)
        if score > best_score:
            best_score = score
            category = cat
    return is_bug_fix, category

for commit in repo.iter_commits(max_count=50): #no. of commits stored in chromadb
    
    is_bug_fix, fix_category = classify_commit(commit.message)

    files = commit.stats.files
    #print(f"\n{commit.hexsha[:8]} | {commit.message.strip()}")
    #for filepath, stats in files.items():
    #    print(f" {filepath} +{stats['insertions']} -{stats['deletions']}")
    #print(f"{commit.hexsha[:8]} | {commit.author.name} | {commit.authored_datetime} | {commit.message.strip()[:60]}")
    
    #chu
    file_list = ", ".join(files.keys())
    insertions = sum(s['insertions'] for s in files.values())
    deletions = sum(s['deletions'] for s in files.values())

    chunk = (
        f"{commit.message.strip()}\n"
        f"Hash: {commit.hexsha[:8]}\n"
        f"Author: {commit.author.name}\n"
        f"Date: {commit.authored_datetime.strftime('%Y-%m-%d')}\n"
        f"Files: {file_list} (+{insertions}), -{deletions})"
        
    )

    chunks.append(chunk)
    ids.append(commit.hexsha[:8])
    metadatas.append({
        "author": commit.author.name,
        "date": commit.authored_datetime.strftime("%Y-%m_%d"),
        "message": commit.message.strip(),
    })


embeddings = model.encode(chunks).tolist()

collection.add(
    ids = ids,
    documents=chunks,
    embeddings=embeddings,
    metadatas = metadatas,
)

print(f"Stored {len(chunks)} commits in chromadb") 

questions = [
    "what was updated in the readme", 
    "what changes were made in january", 
    "what did tammynpm work on", 
]

for query in questions:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings = query_embedding,
        n_results = 5,
    )    
    
    context = ""
    for i in range(len(results["documents"][0])):
        context += results["documents"][0][i] + "\n\n---\n\n"
    prompt = f"""Based on these git commits, answer the question. 

COMMITS:
{context}

QUESTION: {query}
ANSWER:
"""

    resp = requests.post(
        "http://localhost:11434/api/generate", json={"model": "phi3:mini", "prompt": prompt, "stream": False}, timeout=120
    )

    print(f"\nQ: {query}")
    print(f"A: {resp.json()['response']}\n")
    print("-" * 40)

