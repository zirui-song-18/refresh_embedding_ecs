"""ECS Task: Read embeddings from S3 chunks, write to AOSS via _bulk + Painless.

Runs to completion — no 15-min limit, no Step Function event overhead.
Single invocation handles the entire write-back (reads all chunk files).
Includes adaptive backoff for AOSS throttling.
"""

import io
import json
import logging
import os
import sys
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

    # Only fail hard if ALL writes failed — indicates fundamental issue (permissions, wrong endpoint)
    # Partial failures are handled by the CompletenessCheck + retry loop
    total_processed = total_written + total_noop + total_not_found + total_failures
    if total_processed > 0 and total_written == 0 and total_noop == 0:
        logger.error(f"All {total_failures} writes failed with 0 success — "
                     f"likely a permissions or configuration issue")
        sys.exit(1)


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

        # Retry each sub-batch up to 3 times on failure
        batch_written, batch_noop, batch_not_found, batch_failures = 0, 0, 0, 0
        remaining = sub_records

        for attempt in range(3):
            if not remaining:
                break

            if backoff_delay > 0:
                time.sleep(backoff_delay)

            bulk_lines = []
            for rec in remaining:
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

            bulk_body_inner = "\n".join(bulk_lines) + "\n"

            try:
                bulk_resp = aoss_request(endpoint, "POST", "_bulk", bulk_body_inner)
            except RuntimeError as e:
                backoff_delay = min(backoff_delay + 5, 60)
                if attempt < 2:
                    logger.warning(f"Bulk failed (attempt {attempt+1}/3, {len(remaining)} docs), "
                                   f"backoff={backoff_delay}s: {e}")
                    continue
                else:
                    batch_failures += len(remaining)
                    logger.error(f"Bulk permanently failed ({len(remaining)} docs): {e}")
                    break

            # Parse results and collect failed records for retry
            resp_items = bulk_resp.get("items", [])
            if len(resp_items) != len(remaining):
                logger.warning(f"Bulk response items ({len(resp_items)}) != sent records ({len(remaining)}). "
                               f"Treating {len(remaining) - len(resp_items)} unmatched records as failures.")
                batch_failures += max(0, len(remaining) - len(resp_items))

            failed_in_batch = []
            for idx, item in enumerate(resp_items):
                result = item.get("update", {})
                status = result.get("status", 0)
                if status == 200:
                    if result.get("result") == "noop":
                        batch_noop += 1
                    else:
                        batch_written += 1
                elif status == 404:
                    batch_not_found += 1
                elif status == 429:
                    # Throttled — retry this record
                    failed_in_batch.append(remaining[idx])
                else:
                    batch_failures += 1

            remaining = failed_in_batch
            if remaining:
                backoff_delay = min(backoff_delay + 5, 60)
                logger.warning(f"Batch attempt {attempt+1}: {len(remaining)} throttled, "
                               f"retrying with backoff={backoff_delay}s")
            else:
                backoff_delay = max(backoff_delay - 1, 0)

        # Any still-remaining records are permanent failures
        batch_failures += len(remaining)

        written += batch_written
        noop += batch_noop
        not_found += batch_not_found
        failures += batch_failures

    return written, noop, not_found, failures


if __name__ == "__main__":
    main()
