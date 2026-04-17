"""ECS Task: Read embeddings from S3 chunks, write to AOSS via _bulk + Painless.

Runs to completion — no 15-min limit, no Step Function event overhead.
Single invocation handles the entire write-back (reads all chunk files).
Includes adaptive backoff for AOSS throttling.
"""

import io
import json
import logging
import os
import time

import boto3
from aoss_client import aoss_request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
BULK_BATCH_SIZE = 500  # docs per _bulk request


def main():
    endpoint = os.environ["COLLECTION_ENDPOINT"]
    index_name = os.environ["INDEX_NAME"]
    v2_field = os.environ["V2_FIELD"]
    s3_bucket = os.environ["S3_BUCKET"]
    s3_prefix = os.environ["S3_PREFIX"]

    embeddings_prefix = f"{s3_prefix}/embeddings/"
    logger.info(f"WriteEmbeddings starting: index={index_name}, v2_field={v2_field}")
    logger.info(f"Reading from s3://{s3_bucket}/{embeddings_prefix}")

    # List all embedding chunk files
    chunk_keys = _list_chunks(s3_bucket, embeddings_prefix)
    logger.info(f"Found {len(chunk_keys)} embedding chunk files")

    if not chunk_keys:
        logger.warning("No embedding chunks found!")
        return

    total_written = 0
    total_noop = 0
    total_not_found = 0
    total_failures = 0
    start_time = time.time()

    for i, chunk_key in enumerate(chunk_keys):
        logger.info(f"Processing chunk {i+1}/{len(chunk_keys)}: {chunk_key}")

        # Read chunk from S3
        obj = s3.get_object(Bucket=s3_bucket, Key=chunk_key)
        records = []
        for line in io.TextIOWrapper(obj["Body"], encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

        # Write in sub-batches with adaptive backoff
        w, n, nf, f = _write_with_backoff(endpoint, index_name, v2_field, records)
        total_written += w
        total_noop += n
        total_not_found += nf
        total_failures += f

        elapsed = time.time() - start_time
        total_docs = total_written + total_noop + total_not_found
        rate = total_docs / elapsed if elapsed > 0 else 0
        logger.info(f"Chunk {i+1}/{len(chunk_keys)}: written={w}, noop={n}, failures={f} | "
                     f"Total: {total_docs:,} docs, {rate:.0f} docs/sec")

    elapsed = time.time() - start_time
    logger.info(f"WriteEmbeddings complete: written={total_written:,}, noop={total_noop:,}, "
                f"not_found={total_not_found:,}, failures={total_failures:,}, "
                f"time={elapsed:.0f}s ({(total_written+total_noop)/max(elapsed,1):.0f} docs/sec)")

    # Write summary
    summary = {
        "total_written": total_written,
        "total_noop": total_noop,
        "total_not_found": total_not_found,
        "total_failures": total_failures,
        "elapsed_seconds": int(elapsed),
    }
    s3.put_object(
        Bucket=s3_bucket,
        Key=f"{s3_prefix}/write_summary.json",
        Body=json.dumps(summary).encode("utf-8"),
    )


def _list_chunks(s3_bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl"):
                keys.append(obj["Key"])
    return sorted(keys)


def _write_with_backoff(endpoint, index_name, v2_field, records):
    """Write records with adaptive backoff on throttling."""
    written, noop, not_found, failures = 0, 0, 0, 0
    backoff_delay = 0

    for sub_start in range(0, len(records), BULK_BATCH_SIZE):
        sub_records = records[sub_start:sub_start + BULK_BATCH_SIZE]

        if backoff_delay > 0:
            time.sleep(backoff_delay)

        bulk_lines = []
        for rec in sub_records:
            action = json.dumps({"update": {"_id": rec["doc_id"], "_index": index_name}})
            script = json.dumps({
                "script": {
                    "source": (
                        f"if (ctx._source.containsKey('{v2_field}') == false "
                        f"|| ctx._source.{v2_field} == null) "
                        f"{{ ctx._source.{v2_field} = params.vec }} "
                        f"else {{ ctx.op = 'noop' }}"
                    ),
                    "lang": "painless",
                    "params": {"vec": rec["embedding"]},
                }
            })
            bulk_lines.append(action)
            bulk_lines.append(script)

        bulk_body = "\n".join(bulk_lines) + "\n"

        try:
            bulk_resp = aoss_request(endpoint, "POST", "_bulk", bulk_body)
        except RuntimeError as e:
            # Entire bulk failed — increase backoff
            failures += len(sub_records)
            backoff_delay = min(backoff_delay + 2, 30)
            logger.warning(f"Bulk failed ({len(sub_records)} docs), backoff={backoff_delay}s: {e}")
            continue

        # Parse per-item results
        batch_failures = 0
        for item in bulk_resp.get("items", []):
            result = item.get("update", {})
            status = result.get("status", 0)
            if status == 200:
                if result.get("result") == "noop":
                    noop += 1
                else:
                    written += 1
            elif status == 404:
                not_found += 1
            else:
                failures += 1
                batch_failures += 1

        # Adaptive backoff: increase on failures, decrease on success
        if batch_failures > 0:
            backoff_delay = min(backoff_delay + 1, 30)
        else:
            backoff_delay = max(backoff_delay - 0.5, 0)

    return written, noop, not_found, failures


if __name__ == "__main__":
    main()
