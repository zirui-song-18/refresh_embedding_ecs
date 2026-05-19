"""ECS Task: Read embeddings from S3 chunks, write to AOSS via _bulk + Painless.

Per-chunk 100% guarantee: retries failed docs within each chunk until all
succeed or max retries exhausted. No outer completeness loop needed.

Streams per-chunk: read one S3 chunk → build batches → parallel write → retry
failed → next chunk. Peak memory = one chunk.
"""

import io
import json
import logging
import os
import sys
import time

import boto3
from backoff import AdaptiveBackoff
from bulk_writer import parallel_bulk_write
from encryption import download_and_decrypt_from_metadata, is_encryption_enabled

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
BULK_BATCH_SIZE = 500
_MAX_CHUNK_RETRIES = 10


def main():
    endpoint = os.environ["COLLECTION_ENDPOINT"]
    index_name = os.environ["INDEX_NAME"]
    v2_field = os.environ["V2_FIELD"]
    s3_bucket = os.environ["S3_BUCKET"]
    s3_prefix = os.environ["S3_PREFIX"]
    concurrency = int(os.environ.get("WRITE_CONCURRENCY", "1"))

    embeddings_prefix = f"{s3_prefix}/embeddings/"
    logger.info(f"WriteEmbeddings starting: index={index_name}, v2_field={v2_field}, "
                f"concurrency={concurrency}")

    chunk_keys = _list_chunks(s3_bucket, embeddings_prefix)
    logger.info(f"Found {len(chunk_keys)} embedding chunk files")

    if not chunk_keys:
        logger.warning("No embedding chunks found!")
        return

    total_success = 0
    total_failures = 0
    start_time = time.time()

    for i, chunk_key in enumerate(chunk_keys):
        records = _read_chunk(s3_bucket, chunk_key)
        records_by_id = {rec["doc_id"]: rec for rec in records}

        # Per-chunk guarantee: retry until all records in this chunk succeed
        remaining_records = records
        chunk_retry = 0
        chunk_backoff = AdaptiveBackoff(initial=5.0, max_delay=120.0)

        while remaining_records:
            batches = _build_update_batches(remaining_records, index_name, v2_field)
            success, failed_items = parallel_bulk_write(
                endpoint, batches, concurrency,
                task_label=f"WriteEmbeddings[{i + 1}/{len(chunk_keys)}]"
            )
            total_success += success

            # Extract doc_ids from failed items to retry with original records
            failed_ids = set()
            for item in failed_items:
                action = json.loads(item["action"])
                failed_ids.add(action["update"]["_id"])

            remaining_records = [records_by_id[did] for did in failed_ids if did in records_by_id]

            if remaining_records:
                chunk_retry += 1
                if chunk_retry >= _MAX_CHUNK_RETRIES:
                    logger.error(
                        f"Chunk {i + 1}: {len(remaining_records)} docs failed after "
                        f"{_MAX_CHUNK_RETRIES} chunk retries — giving up"
                    )
                    total_failures += len(remaining_records)
                    remaining_records = []
                else:
                    logger.warning(
                        f"Chunk {i + 1}: {len(remaining_records)} docs need chunk-level retry "
                        f"(round {chunk_retry}/{_MAX_CHUNK_RETRIES})"
                    )
                    chunk_backoff.on_failure()
                    chunk_backoff.wait()

        del records, records_by_id  # free memory

        elapsed = time.time() - start_time
        rate = total_success / elapsed if elapsed > 0 else 0
        logger.info(f"Chunk {i + 1}/{len(chunk_keys)} complete | "
                    f"Total: success={total_success:,}, failures={total_failures}, "
                    f"retries={chunk_retry}, {rate:.0f} docs/sec")

    elapsed = time.time() - start_time
    logger.info(f"WriteEmbeddings complete: success={total_success:,}, failures={total_failures:,}, "
                f"time={elapsed:.0f}s")

    s3.put_object(
        Bucket=s3_bucket,
        Key=f"{s3_prefix}/write_summary.json",
        Body=json.dumps({
            "total_written": total_success,
            "total_failures": total_failures,
            "elapsed_seconds": int(elapsed),
        }).encode("utf-8"),
    )

    total_processed = total_success + total_failures
    if total_processed > 0 and total_success == 0:
        logger.error(f"All {total_failures} writes failed — likely a permissions or configuration issue")
        sys.exit(1)


def _read_chunk(s3_bucket, chunk_key):
    """Read one S3 chunk file, return list of record dicts."""
    if is_encryption_enabled():
        plaintext = download_and_decrypt_from_metadata(s3, s3_bucket, chunk_key)
        lines_raw = plaintext.decode("utf-8").splitlines()
    else:
        obj = s3.get_object(Bucket=s3_bucket, Key=chunk_key)
        lines_raw = io.TextIOWrapper(obj["Body"], encoding="utf-8").readlines()

    records = []
    for line in lines_raw:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Skipping non-JSON line: {line[:100]}")
            continue
        if "doc_id" not in rec or "embedding" not in rec:
            logger.warning(f"Skipping malformed record: {line[:100]}")
            continue
        records.append(rec)
    return records


def _build_update_batches(records, index_name, v2_field):
    """Convert records into batches for parallel_bulk_write.

    Uses params.field with [] notation to prevent Painless injection.
    """
    batches = []
    current_batch = []

    for rec in records:
        action = json.dumps({"update": {"_id": rec["doc_id"], "_index": index_name}})
        body = json.dumps({
            "script": {
                "source": (
                    "if (ctx._source.containsKey(params.field) == false "
                    "|| ctx._source[params.field] == null) "
                    "{ ctx._source[params.field] = params.vec } "
                    "else { ctx.op = 'noop' }"
                ),
                "lang": "painless",
                "params": {"vec": rec["embedding"], "field": v2_field},
            }
        })
        current_batch.append({"action": action, "body": body})

        if len(current_batch) >= BULK_BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches


def _list_chunks(s3_bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl"):
                keys.append(obj["Key"])
    return sorted(keys)


if __name__ == "__main__":
    main()
