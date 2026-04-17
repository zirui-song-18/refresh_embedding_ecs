#!/usr/bin/env python3
"""Set up a test index with sample documents for the dual-field reindex POC.

Computes embeddings via SageMaker directly (bypasses ingest pipeline) to avoid
connector pre_process_function configuration complexity.

Usage:
    python setup_test_index.py \
        --endpoint https://xxx.us-west-2.aoss.amazonaws.com \
        --index test-reindex \
        --model-id YOUR_MODEL_ID \
        --sagemaker-endpoint SM_ENDPOINT_NAME \
        --num-docs 200
"""

import argparse
import json
import sys
import time

import boto3
from requests_aws4auth import AWS4Auth
import requests

# --- Auth setup ---
session = boto3.Session(region_name="us-west-2")
creds = session.get_credentials().get_frozen_credentials()
auth = AWS4Auth(creds.access_key, creds.secret_key, "us-west-2", "aoss",
                session_token=creds.token)
sm_runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")


def req(method, endpoint, path, body=None):
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body) if body else None
    if isinstance(body, str):
        headers["Content-Type"] = "application/x-ndjson"
        data = body
    r = requests.request(method, url, auth=auth, headers=headers, data=data)
    if r.status_code >= 400:
        print(f"ERROR {r.status_code}: {r.text[:500]}")
    return r.json() if r.text else {}


def get_embeddings(sm_endpoint_name, texts):
    payload = json.dumps({"inputs": texts}).encode("utf-8")
    response = sm_runtime.invoke_endpoint(
        EndpointName=sm_endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            return result
        return [result]
    raise ValueError(f"Unexpected response: {type(result)}")


SAMPLE_TEXTS = [
    "OpenSearch is a distributed search and analytics engine based on Apache Lucene.",
    "Amazon Web Services provides cloud computing services to businesses worldwide.",
    "Machine learning models can be fine-tuned on domain-specific data for better performance.",
    "Kubernetes is an open-source container orchestration platform for automating deployment.",
    "Natural language processing enables computers to understand and generate human language.",
    "Vector databases store high-dimensional embeddings for similarity search applications.",
    "The transformer architecture revolutionized deep learning for sequence modeling tasks.",
    "Serverless computing eliminates the need to manage infrastructure for application deployment.",
    "Graph neural networks capture relationships between entities in structured data.",
    "Reinforcement learning trains agents through reward signals in interactive environments.",
    "Data pipelines automate the extraction, transformation, and loading of information.",
    "Microservices architecture decomposes applications into small independently deployable services.",
    "Semantic search uses meaning rather than keywords to find relevant documents.",
    "Federated learning trains models across decentralized data sources without sharing raw data.",
    "Embedding models convert text into dense vector representations for similarity comparison.",
    "API gateways manage traffic routing, authentication, and rate limiting for backend services.",
    "Time series databases optimize storage and queries for temporal data patterns.",
    "Convolutional neural networks excel at extracting features from images and spatial data.",
    "Infrastructure as code manages cloud resources through version-controlled configuration files.",
    "Attention mechanisms allow models to focus on relevant parts of input sequences.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="AOSS collection endpoint")
    parser.add_argument("--index", default="test-reindex", help="Index name")
    parser.add_argument("--model-id", required=True, help="ML model ID in AOSS (for pipeline config)")
    parser.add_argument("--sagemaker-endpoint", required=True, help="SageMaker endpoint name")
    parser.add_argument("--pipeline-name", default="test-reindex-pipeline")
    parser.add_argument("--num-docs", type=int, default=200, help="Number of test docs")
    parser.add_argument("--dimension", type=int, default=384)
    args = parser.parse_args()

    print(f"=== Setting up test index: {args.index} ===\n")

    # 1. Create index (NO default pipeline — we compute embeddings externally)
    print(f"1. Creating index {args.index} (no default pipeline)...")
    resp = req("PUT", args.endpoint, args.index, {
        "settings": {
            "index.knn": True,
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "search_text": {"type": "knn_vector", "dimension": args.dimension},
            }
        },
    })
    print(f"   Index: {resp}\n")

    # 2. Bulk-index documents WITH pre-computed embeddings
    print(f"2. Indexing {args.num_docs} documents (computing embeddings via SageMaker)...")
    batch_size = 20
    total_indexed = 0

    for batch_start in range(0, args.num_docs, batch_size):
        batch_end = min(batch_start + batch_size, args.num_docs)

        # Prepare texts
        texts = []
        for i in range(batch_start, batch_end):
            text = f"[Doc {i}] {SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]}"
            texts.append(text)

        # Compute embeddings via SageMaker
        try:
            embeddings = get_embeddings(args.sagemaker_endpoint, texts)
        except Exception as e:
            print(f"   Batch {batch_start}-{batch_end}: SageMaker error: {e}")
            continue

        # Build bulk body with text + pre-computed embedding
        bulk_lines = []
        for text, emb in zip(texts, embeddings):
            bulk_lines.append(json.dumps({"index": {"_index": args.index}}))
            bulk_lines.append(json.dumps({"text": text, "search_text": emb}))

        bulk_body = "\n".join(bulk_lines) + "\n"
        resp = req("POST", args.endpoint, "_bulk", bulk_body)
        errors = resp.get("errors", False)
        items_ok = sum(1 for item in resp.get("items", []) if item.get("index", {}).get("status") in (200, 201))
        total_indexed += items_ok
        print(f"   Batch {batch_start}-{batch_end}: indexed={items_ok}, errors={errors}")

    # 3. Wait for AOSS refresh
    print(f"\n3. Waiting 65s for AOSS refresh...")
    time.sleep(65)

    # 4. Verify
    print("4. Verifying...")
    count_resp = req("POST", args.endpoint, f"{args.index}/_count", {"query": {"match_all": {}}})
    total = count_resp.get("count", 0)
    print(f"   Total docs: {total}")

    has_emb_resp = req("POST", args.endpoint, f"{args.index}/_count", {
        "query": {"exists": {"field": "search_text"}}
    })
    has_emb = has_emb_resp.get("count", 0)
    print(f"   Docs with search_text embedding: {has_emb}")

    search_resp = req("POST", args.endpoint, f"{args.index}/_search", {
        "size": 1,
        "_source": ["text", "search_text"],
        "query": {"match_all": {}},
    })
    hits = search_resp.get("hits", {}).get("hits", [])
    if hits:
        doc = hits[0]["_source"]
        v1 = doc.get("search_text")
        print(f"   Sample doc: {doc.get('text', 'N/A')[:60]}...")
        if v1:
            print(f"   Embedding: [{v1[0]:.6f}, {v1[1]:.6f}, ...] (dim={len(v1)})")
        else:
            print(f"   Embedding: MISSING")

    print(f"\n=== Setup complete ===")
    print(f"Index: {args.index}")
    print(f"Docs: {total} ({has_emb} with embeddings)")
    print(f"Model ID: {args.model_id}")
    print(f"SageMaker endpoint: {args.sagemaker_endpoint}")
    print(f"\nNext: get State Machine ARN:")
    print(f"  aws cloudformation describe-stacks --stack-name poc-dual-field-reindex --region us-west-2 --query \"Stacks[0].Outputs[?OutputKey=='StateMachineArn'].OutputValue\" --output text")
    print(f"\nThen trigger reindex:")
    print(f"  python tests/trigger_reindex.py \\")
    print(f"    --endpoint {args.endpoint} --index {args.index} \\")
    print(f"    --model-id {args.model_id} --pipeline-name {args.pipeline_name} \\")
    print(f"    --sagemaker-endpoint {args.sagemaker_endpoint} \\")
    print(f"    --state-machine-arn <ARN>")


if __name__ == "__main__":
    main()
