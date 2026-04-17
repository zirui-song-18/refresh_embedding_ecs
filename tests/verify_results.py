#!/usr/bin/env python3
"""Verify dual-field reindex results.

Usage:
    python verify_results.py \
        --endpoint https://xxx.us-west-2.aoss.amazonaws.com \
        --index test-reindex
"""

import argparse
import json

import boto3
from requests_aws4auth import AWS4Auth
import requests

session = boto3.Session(region_name="us-west-2")
creds = session.get_credentials().get_frozen_credentials()
auth = AWS4Auth(creds.access_key, creds.secret_key, "us-west-2", "aoss",
                session_token=creds.token)


def req(method, endpoint, path, body=None):
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body) if body else None
    r = requests.request(method, url, auth=auth, headers=headers, data=data)
    return r.json() if r.text else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--index", default="test-reindex")
    parser.add_argument("--v1-field", default="search_text")
    parser.add_argument("--v2-field", default="search_text_v2")
    args = parser.parse_args()

    print(f"=== Verifying reindex results for {args.index} ===\n")

    # 1. Total docs
    total = req("POST", args.endpoint, f"{args.index}/_count",
                {"query": {"match_all": {}}})["count"]
    print(f"1. Total documents: {total}")

    # 2. Docs with v1
    has_v1 = req("POST", args.endpoint, f"{args.index}/_count", {
        "query": {"exists": {"field": args.v1_field}}
    })["count"]
    print(f"2. Docs with {args.v1_field}: {has_v1}")

    # 3. Docs with v2
    has_v2 = req("POST", args.endpoint, f"{args.index}/_count", {
        "query": {"exists": {"field": args.v2_field}}
    })["count"]
    print(f"3. Docs with {args.v2_field}: {has_v2}")

    # 4. Docs missing v2
    missing_v2 = req("POST", args.endpoint, f"{args.index}/_count", {
        "query": {"bool": {"must_not": {"exists": {"field": args.v2_field}}}}
    })["count"]
    print(f"4. Docs missing {args.v2_field}: {missing_v2}")

    # 5. Completion percentage
    pct = round(has_v2 / max(total, 1) * 100, 2)
    print(f"5. Completion: {pct}%")

    # 6. Sample doc
    sample = req("POST", args.endpoint, f"{args.index}/_search", {
        "size": 1,
        "_source": ["text", args.v1_field, args.v2_field],
        "query": {"exists": {"field": args.v2_field}},
    })
    hits = sample.get("hits", {}).get("hits", [])
    if hits:
        doc = hits[0]["_source"]
        print(f"\n6. Sample document:")
        print(f"   text: {doc.get('text', 'N/A')[:80]}...")
        v1 = doc.get(args.v1_field)
        v2 = doc.get(args.v2_field)
        print(f"   {args.v1_field}: [{v1[0]:.6f}, {v1[1]:.6f}, ...] (dim={len(v1)})" if v1 else f"   {args.v1_field}: None")
        print(f"   {args.v2_field}: [{v2[0]:.6f}, {v2[1]:.6f}, ...] (dim={len(v2)})" if v2 else f"   {args.v2_field}: None")

    # 7. KNN search on v2
    if has_v2 > 0 and hits:
        v2_vec = hits[0]["_source"].get(args.v2_field)
        if v2_vec:
            knn_resp = req("POST", args.endpoint, f"{args.index}/_search", {
                "size": 3,
                "_source": ["text"],
                "query": {
                    "knn": {
                        args.v2_field: {
                            "vector": v2_vec[:384],
                            "k": 3,
                        }
                    }
                },
            })
            knn_hits = knn_resp.get("hits", {}).get("hits", [])
            print(f"\n7. KNN search on {args.v2_field} (k=3):")
            for i, h in enumerate(knn_hits):
                print(f"   [{i+1}] score={h['_score']:.4f} | {h['_source']['text'][:70]}...")

    # Summary
    print(f"\n=== Summary ===")
    if missing_v2 == 0 and total > 0:
        print(f"PASS: All {total} docs have {args.v2_field}. Reindex complete.")
    elif total == 0:
        print(f"WARN: Index is empty.")
    else:
        print(f"INCOMPLETE: {missing_v2}/{total} docs still missing {args.v2_field}.")


if __name__ == "__main__":
    main()
