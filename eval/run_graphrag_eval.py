#!/usr/bin/env python3
"""Unified GraphRAG evaluation for LogicPoison reproduction.
Tests all 3 frameworks on the poisoned corpus with deepseek-v4-flash."""

import json, os, sys, re, time, random
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY", "sk-jvEg4M4o2eYqFNkefEhpZrP4NpyRn3r56P2NjVrpdVejKbOq1Bmk1xL3eV08oqMV")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://opencode.ai/zen/go/v1")
MODEL = "deepseek-v4-flash"
N_SAMPLES = 10

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def norm(s):
    if not s: return ''
    s = str(s).lower(); s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def load_data(dataset_name):
    answers = {}
    with open(f'datasets/{dataset_name}/answers.jsonl') as f:
        for l in f:
            if l.strip():
                r = json.loads(l.strip()); answers[r['id']] = r['answer']
    queries = []
    with open(f'results/poisoned_data/{dataset_name}/queries.jsonl') as f:
        for l in f:
            if l.strip(): queries.append(json.loads(l.strip()))
    corpus = []
    with open(f'results/poisoned_data/{dataset_name}/corpus.jsonl') as f:
        for l in f:
            if l.strip(): corpus.append(json.loads(l.strip()))
    return answers, queries, corpus

def generate_answer(query, contexts):
    ctx = '\n\n'.join([f'[{i+1}] {c.get("title","")}: {c.get("text","")[:300]}' for i,c in enumerate(contexts)])
    prompt = f'Answer briefly based on context:\n\n{ctx}\n\nQuestion: {query}\n\nAnswer:'
    for max_tok in [2000, 4000, 8000]:
        rsp = client.chat.completions.create(model=MODEL, messages=[{'role':'user','content':prompt}], temperature=0, max_tokens=max_tok)
        content = rsp.choices[0].message.content or ''
        if content.strip(): return content.strip()
        if rsp.choices[0].finish_reason != 'length': break
    return ''

def compute_asr(predictions, answers):
    total = wrong = 0
    for p in predictions:
        pid = p['id']
        if pid not in answers: continue
        total += 1
        if not norm(answers[pid]) in norm(p['output']): wrong += 1
    return wrong/total*100 if total else 0, total

def run_framework(name, retrieve_fn, dataset='hotpotqa'):
    print(f'\n=== {name} on {dataset} ===')
    answers, queries, corpus = load_data(dataset)
    random.seed(42); random.shuffle(queries)
    queries = queries[:N_SAMPLES]
    
    preds = []
    for q in queries:
        qid = str(q.get('_id', q.get('id', '')))
        qtext = q.get('text', q.get('question', ''))
        ctxs = retrieve_fn(qtext, corpus)
        ans = generate_answer(qtext, ctxs)
        preds.append({'id': qid, 'output': ans, 'question': qtext})
        time.sleep(0.1)
    
    asr, total = compute_asr(preds, answers)
    print(f'{name} ASR: {asr:.1f}% ({total} samples)')
    
    with open(f'results/poisoned_data/{dataset}/{name.lower()}_predictions.jsonl', 'w') as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False)+'\n')
    return asr

def run_all():
    results = {}
    
    from run_naive_rag import load_retriever, build_index, retrieve as contriever_retrieve
    
    embed_fn = load_retriever()
    docs, index = build_index(embed_fn, 'results/poisoned_data/hotpotqa/corpus.jsonl')
    
    def contr_retrieve(q, corpus):
        import numpy as np
        q_emb = embed_fn([q]).astype(np.float32)
        return contriever_retrieve(index, docs, q_emb, k=3)
    
    results['NaiveRAG'] = run_framework('NaiveRAG', contr_retrieve)
    
    try:
        from hipporag import HippoRAG
        hippo = HippoRAG(save_dir='/tmp/hippo_eval', llm_model_name=MODEL, llm_base_url=BASE_URL)
        corpus_texts = []
        with open('results/poisoned_data/hotpotqa/corpus.jsonl') as f:
            for l in f:
                if l.strip(): corpus_texts.append(json.loads(l.strip()).get('text', ''))
        hippo.index(docs=corpus_texts[:100])
        
        def hippo_retrieve(q, corpus):
            result = hippo.retrieve(queries=[q], num_to_retrieve=3)
            return [{'text': r} for r in result.get('passages', [])[:3]]
        
        results['HippoRAG2'] = run_framework('HippoRAG2', hippo_retrieve)
    except Exception as e:
        print(f'HippoRAG2: blocked - {e}')
        results['HippoRAG2'] = 'blocked'
    
    try:
        from graphrag.query.api import global_search, local_search
        results['GraphRAG'] = 'needs indexing (v3 API change)'
    except Exception as e:
        results['GraphRAG'] = f'blocked: {e}'
    
    try:
        from gfmrag import GFMRetriever
        results['GFM-RAG'] = 'needs graph construction'
    except Exception as e:
        results['GFM-RAG'] = f'blocked: {e}'
    
    print('\n=== FINAL ===')
    for name, r in results.items():
        print(f'{name}: {r}')

if __name__ == '__main__':
    run_all()
