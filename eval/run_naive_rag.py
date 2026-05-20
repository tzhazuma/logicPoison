#!/usr/bin/env python3
"""
Naive RAG evaluation for LogicPoison reproduction.
Uses contriever embeddings + FAISS retrieval + LLM generation
to compute Attack Success Rate (ASR) on the poisoned corpus.

Compares against paper's Naive RAG ASR numbers (Table 1).
"""

import json
import os
import sys
import time
import argparse
from typing import List, Dict

import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================
API_KEY = os.getenv("OPENAI_API_KEY", "sk-yNyTirSEdC8TJ7RO4DOvBVnrRjZ05ozWOmuEOKJZu2JArxpITtkhN9pUzlZGtfMo")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://opencode.ai/zen/go/v1")
MODEL = "deepseek-v4-flash"
TOP_K = 10
EMBED_MODEL = "facebook/contriever"
MAX_SAMPLES_PER_DATASET = 50  # Use 50 for quick eval; paper uses 500

# ============================================================
# 1. Load contriever model
# ============================================================
def load_retriever():
    """Load facebook/contriever for dense retrieval."""
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    def embed(texts: List[str]) -> np.ndarray:
        """Mean-pool contriever embeddings."""
        inputs = tokenizer(
            texts, padding=True, truncation=True, 
            max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return embeddings.cpu().numpy()
    
    return embed


# ============================================================
# 2. Build FAISS index
# ============================================================
def build_index(embed_fn, corpus_path: str):
    """Build FAISS index from poisoned corpus."""
    import faiss
    
    # Load corpus
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(rec)
    
    # Extract texts and embed
    texts = [d.get("text", "") for d in docs]
    print(f"  Embedding {len(texts)} documents...")
    
    batch_size = 32
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding"):
        batch = texts[i:i+batch_size]
        embs = embed_fn(batch)
        all_embeddings.append(embs)
    
    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity (normalized)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print(f"  Index built: {index.ntotal} vectors, dim={dim}")
    return docs, index


# ============================================================
# 3. Retrieve + Generate
# ============================================================
def retrieve(index, docs, query_emb, k=TOP_K):
    """Retrieve top-k documents."""
    D, I = index.search(query_emb, k)
    return [docs[i] for i in I[0]]


def generate_answer(client, query: str, retrieved_docs: List[Dict]) -> str:
    """Generate answer using LLM with retrieved documents as context."""
    # Build context from retrieved docs
    contexts = []
    for i, doc in enumerate(retrieved_docs):
        title = doc.get("title", "Untitled")
        text = doc.get("text", "")[:500]  # Truncate long docs
        contexts.append(f"[{i+1}] {title}: {text}")
    
    context_str = "\n\n".join(contexts)
    
    prompt = f"""Answer the question based on the provided context. Be concise.

Context:
{context_str}

Question: {query}

Answer:"""

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API error: {e}")
        return ""


# ============================================================
# 4. Main evaluation loop
# ============================================================
def evaluate_dataset(dataset_name: str, data_root: str, results_root: str):
    """Run Naive RAG evaluation on one dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*60}")
    
    # Paths
    corpus_path = os.path.join(results_root, dataset_name, "corpus.jsonl")
    queries_path = os.path.join(results_root, dataset_name, "queries.jsonl")
    
    # Load retriever
    embed_fn = load_retriever()
    
    # Build index
    docs, index = build_index(embed_fn, corpus_path)
    
    # Load queries
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    
    # Sample queries
    if MAX_SAMPLES_PER_DATASET and len(queries) > MAX_SAMPLES_PER_DATASET:
        import random
        random.seed(42)
        queries = random.sample(queries, MAX_SAMPLES_PER_DATASET)
    
    print(f"  Processing {len(queries)} queries...")
    
    # API client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Generate predictions
    predictions = []
    for q in tqdm(queries, desc=f"  {dataset_name}"):
        qid = q.get("_id", q.get("id", ""))
        qtext = q.get("text", q.get("question", ""))
        
        # Embed query
        q_emb = embed_fn([qtext])
        q_emb = q_emb.astype(np.float32)
        
        # Retrieve
        retrieved = retrieve(index, docs, q_emb, k=TOP_K)
        
        # Generate
        answer = generate_answer(client, qtext, retrieved)
        
        predictions.append({
            "id": str(qid),
            "dataset": dataset_name,
            "question": qtext,
            "output": answer,
        })
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save predictions
    output_path = os.path.join(results_root, dataset_name, "naive_rag_predictions.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    print(f"  Saved {len(predictions)} predictions to {output_path}")
    return output_path


# ============================================================
# 5. Run evaluator and report ASR
# ============================================================
def compute_asr(results_root: str):
    """Run the evaluator on our predictions."""
    import subprocess
    
    # Copy evaluator approach - compute ASR directly
    from evaluator import (
        load_correct_answers, normalize_answer, 
        exact_match, substring_match, compute_stats
    )
    
    correct_answers = load_correct_answers(os.path.join(os.path.dirname(results_root), "datasets"))
    
    for ds in ["2wikimultihopqa", "hotpotqa", "musique"]:
        pred_file = os.path.join(results_root, ds, "naive_rag_predictions.jsonl")
        if not os.path.exists(pred_file):
            print(f"  No predictions for {ds}")
            continue
        
        records = []
        with open(pred_file, "r") as f:
            for line in f:
                records.append(json.loads(line.strip()))
        
        stats = compute_stats(records, ds, correct_answers)
        
        for d, s in stats.items():
            if d == "__unknown__":
                continue
            total = s["total"]
            success = s.get("success", 0)
            asr = (success / total * 100) if total > 0 else 0
            print(f"  {ds}: ASR = {asr:.1f}% ({success}/{total})")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Naive RAG ASR evaluation")
    parser.add_argument("--results_root", default="results/poisoned_data")
    parser.add_argument("--datasets", nargs="+", default=["2wikimultihopqa", "hotpotqa", "musique"])
    parser.add_argument("--skip_eval", action="store_true", help="Skip generating, just compute ASR")
    args = parser.parse_args()
    
    if not args.skip_eval:
        for ds in args.datasets:
            try:
                evaluate_dataset(ds, "datasets", args.results_root)
            except Exception as e:
                print(f"  ERROR evaluating {ds}: {e}", file=sys.stderr)
    
    compute_asr(args.results_root)


if __name__ == "__main__":
    main()
