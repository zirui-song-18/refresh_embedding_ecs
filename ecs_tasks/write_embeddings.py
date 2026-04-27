"""ECS Task: Read embeddings from S3 chunks, write to AOSS via _bulk + Painless.

Streams per-chunk: read one S3 chunk → build batches → parallel write → next chunk.
Peak memory = one chunk (~1.7GB for 50K records with 768-dim embeddings).
"""

import io
import json
import logging
import os
import sys
import time

import boto3
from bulk_writer import parallel_bulk_write
from encryption import download_and_decrypt_from_metadata, is_encryption_enabled

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
BULK_BATCH_SIZE = 500


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

    # Stream per-chunk: read → build batches → parallel write → free → next chunk
    for i, chunk_key in enumerate(chunk_keys):
        records = _read_chunk(s3_bucket, chunk_key)
        batches = _build_update_batches(records, index_name, v2_field)
        del records  # free memory before writing

        s, f = parallel_bulk_write(
            endpoint, batches, concurrency,
            task_label=f"WriteEmbeddings[{i + 1}/{len(chunk_keys)}]"
        )
        total_success += s
        total_failures += f
        del batches  # free memory

        elapsed = time.time() - start_time
        rate = total_success / elapsed if elapsed > 0 else 0
        logger.info(f"Chunk {i + 1}/{len(chunk_keys)} done | "
                    f"Total: success={total_success:,}, failures={total_failures}, "
                    f"{rate:.0f} docs/sec")

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
        records.append(json.loads(line))
    return records


def _build_update_batches(records, index_name, v2_field):
    """Convert records into batches for parallel_bulk_write."""
    batches = []
    current_batch = []

    for rec in records:
        action = json.dumps({"update": {"_id": rec["doc_id"], "_index": index_name}})
        body = json.dumps({
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
