"""
Proofline AI - RAG Evaluation and Benchmark Suite
Runs representative questions across depth modes, measures:
- TTFT (Time to first token)
- Generation latency & Total request time
- Token count & generation speed (tokens/sec)
- Citation accuracy & source validity
- Factual grounding and anti-hallucination
- Model memory and residency stats
"""

import json
import re
import time
import requests

API_URL = "http://127.0.0.1:8000/api/chat/stream"
HEALTH_URL = "http://127.0.0.1:8000/api/health"
AUDIT_URL = "http://127.0.0.1:8000/api/chat/audit"

QUESTIONS = [
    {
        "id": "Q1_bottom_up",
        "category": "Core Conceptual Depth",
        "question": "Explain bottom-up thinking and why it avoids repeated computation.",
        "depth_mode": "detailed",
    },
    {
        "id": "Q2_traversals_comparison",
        "category": "Comparative Analysis & Trade-offs",
        "question": "Compare recursive and iterative traversals, including their trade-offs.",
        "depth_mode": "standard",
    },
    {
        "id": "Q3_diameter_cpp",
        "category": "Algorithmic Process & Code Implementation",
        "question": "Explain how to solve the diameter of a binary tree, with a C++ example.",
        "depth_mode": "detailed",
    },
    {
        "id": "Q4_multi_pdf",
        "category": "Multi-Document Synthesis",
        "question": "What is the difference between Morris Inorder Traversal and Morris Preorder Traversal, and how do threaded binary trees restore modified pointers?",
        "depth_mode": "standard",
    },
    {
        "id": "Q5_absent_concept",
        "category": "Groundedness / Anti-Hallucination",
        "question": "Explain how Dijkstra's algorithm finds the shortest path in a weighted graph.",
        "depth_mode": "brief",
    },
]

# Additional mode comparison test set (same question across 3 modes)
MODE_COMPARISON_QUESTION = "Explain bottom-up thinking and why it avoids repeated computation."


def query_stream(question: str, depth_mode: str = "standard", model: str = "gemma4:e2b"):
    payload = {
        "question": question,
        "depth_mode": depth_mode,
        "model": model,
        "history": [],
    }
    
    t0 = time.perf_counter()
    resp = requests.post(API_URL, json=payload, stream=True, timeout=180)
    resp.raise_for_status()

    sources = []
    tokens = []
    first_token_time = None
    done_payload = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event: sources"):
            pass
        elif line.startswith("data: ") and not sources and "sources" in line:
            data = json.loads(line[6:])
            sources = data.get("sources", [])
        elif line.startswith("event: token"):
            pass
        elif line.startswith("data: ") and "delta" in line:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            data = json.loads(line[6:])
            tokens.append(data.get("delta", ""))
        elif line.startswith("event: done"):
            pass
        elif line.startswith("data: ") and "used_labels" in line:
            done_payload = json.loads(line[6:])

    t1 = time.perf_counter()
    full_answer = "".join(tokens)
    ttft_s = (first_token_time - t0) if first_token_time else 0.0
    total_s = t1 - t0

    # Extract citations in answer
    citations = re.findall(r"\[S\d+\]", full_answer)

    return {
        "question": question,
        "depth_mode": depth_mode,
        "model": model,
        "sources_count": len(sources),
        "sources": sources,
        "tokens_streamed": len(tokens),
        "char_count": len(full_answer),
        "ttft_sec": round(ttft_s, 3),
        "total_sec": round(total_s, 3),
        "server_metrics": done_payload.get("metrics", {}) if done_payload else {},
        "used_labels": done_payload.get("used_labels", []) if done_payload else [],
        "invalid_labels": done_payload.get("invalid_labels", []) if done_payload else [],
        "citations_found": sorted(list(set(citations))),
        "answer": full_answer,
    }


def run_benchmarks(model_name: str = "gemma4:e2b"):
    print(f"=== Starting Repeatable RAG Quality & Performance Benchmark for {model_name} ===")
    
    # Check health first
    h = requests.get(HEALTH_URL).json()
    print(f"Server Health: {h['status']} | Chunks: {h['indexed_chunks']} | Top-K: {h['top_k']} | Num-Ctx: {h['num_ctx']}")
    print("-" * 70)

    results = []

    # 1. Mode depth scaling test
    print("\n--- 1. Depth Mode Comparison (Brief vs. Standard vs. Detailed) ---")
    depth_results = {}
    for mode in ["brief", "standard", "detailed"]:
        print(f"Testing mode={mode} on: '{MODE_COMPARISON_QUESTION[:40]}...'")
        res = query_stream(MODE_COMPARISON_QUESTION, depth_mode=mode, model=model_name)
        depth_results[mode] = {
            "tokens": res["tokens_streamed"],
            "chars": res["char_count"],
            "ttft_sec": res["ttft_sec"],
            "total_sec": res["total_sec"],
            "tps": res["server_metrics"].get("tokens_per_sec", 0),
            "citations": res["citations_found"],
            "answer_preview": res["answer"][:180].replace("\n", " "),
        }
        print(f"  -> {mode.upper()}: {res['tokens_streamed']} tokens, {res['char_count']} chars, TTFT: {res['ttft_sec']}s, Total: {res['total_sec']}s, Speed: {res['server_metrics'].get('tokens_per_sec', 0)} tok/s")

    # 2. Five canonical evaluation questions
    print("\n--- 2. Five Canonical Evaluation Questions ---")
    for q in QUESTIONS:
        print(f"\nEvaluating [{q['id']}] ({q['category']}) in {q['depth_mode'].upper()} mode...")
        print(f"Q: {q['question']}")
        res = query_stream(q["question"], depth_mode=q["depth_mode"], model=model_name)
        res["id"] = q["id"]
        res["category"] = q["category"]
        results.append(res)

        print(f"  Results:")
        print(f"    - Sources retrieved: {res['sources_count']}")
        print(f"    - TTFT: {res['ttft_sec']}s")
        print(f"    - Total Latency: {res['total_sec']}s")
        print(f"    - Streamed Tokens: {res['tokens_streamed']}")
        print(f"    - Generation Speed: {res['server_metrics'].get('tokens_per_sec', 0)} tok/s")
        print(f"    - Citations used: {res['used_labels']} (Invalid: {res['invalid_labels']})")
        print(f"    - Answer snippet:\n      {res['answer'][:250]}...")

    output_file = f"benchmark_results_{model_name.replace(':', '_')}.json"
    full_report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server_config": h,
        "depth_mode_scaling": depth_results,
        "evaluation_questions": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\nAll benchmark results successfully recorded to {output_file}")
    return full_report


if __name__ == "__main__":
    run_benchmarks("gemma4:e2b")
