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
        score=sum(1 for s in signals if s in text)
        if score > best_score:
            best_score = score
            category = cat
    return is_bug_fix, category

def extract_components(files):
    components = set() #to avoid duplicates
    for filepath in files:
        parts = filepath.split("/")
        if len(parts) >=2 :
            first_two = parts[:2]
            joined = "/".join(first_two)
            components.add(joined)
        else:
            components.add(parts[0])
    return sorted(components)

for commit in repo.iter_commits(max_count=50): #no. of commits stored in chromadb
    
    is_bug_fix, fix_category = classify_commit(commit.message)
    
    files = list(commit.stats.files.keys())
    components = extract_components(files)
    insertions = sum(s['insertions'] for s in commit.stats.files.values())
    deletions = sum(s['deletions'] for s in commit.stats.files.values())
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
        "timestamp": int(commit.authored_datetime.timestamp()),
        "subject": commit.message.strip().split("\n")[0],
        "is_bug_fix": is_bug_fix,
        "fix_category": fix_category,
        "components": ",".join(components),
        "files_changed": files_changed,
        "insertions": insertions,
        "deletions": deletions,

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

