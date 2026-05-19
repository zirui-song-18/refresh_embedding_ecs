"""ECS Task: Extract all document texts from AOSS, write to S3 as JSONL chunks.

Extracts all documents that have the text field populated (no v2_field filter).
Runs to completion — no 15-min limit, no Step Function event overhead.
Single invocation handles the entire extraction (PIT + search_after loop).
"""

import json
import logging
import os
import time

import boto3
from aoss_client import aoss_request
from encryption import encrypt_and_upload_with_nonce_prefix, is_encryption_enabled

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
SEARCH_PAGE_SIZE = int(os.environ.get("SEARCH_PAGE_SIZE", "10000"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "50000"))


def main():
    endpoint = os.environ["COLLECTION_ENDPOINT"]
    index_name = os.environ["INDEX_NAME"]
    text_field = os.environ["TEXT_FIELD"]
    s3_bucket = os.environ["S3_BUCKET"]
    s3_prefix = os.environ["S3_PREFIX"]

    logger.info(f"ExtractTexts starting: index={index_name}, text_field={text_field}")
    logger.info(f"Output: s3://{s3_bucket}/{s3_prefix}/texts/")

    # Clean up old text data from previous round
    _cleanup_s3_prefix(s3_bucket, f"{s3_prefix}/texts/")
    logger.info("Cleaned up old texts/ from S3")

    # Create PIT
    pit_resp = aoss_request(
        endpoint, "POST",
        f"{index_name}/_search/point_in_time?keep_alive=15m",
    )
    pit_id = pit_resp["pit_id"]
    logger.info("PIT created")

    total_extracted = 0
    chunk_num = 0
    chunk_buffer = []
    search_after = None
    start_time = time.time()

    try:
        while True:
            # Search for docs that have the text field populated
            search_body = {
                "size": SEARCH_PAGE_SIZE,
                "_source": [text_field],
                "query": {"exists": {"field": text_field}},
                "pit": {"id": pit_id, "keep_alive": "15m"},
                "sort": [{"_doc": "asc"}, {"_id": "asc"}],
            }
            if search_after:
                search_body["search_after"] = search_after

            search_resp = aoss_request(endpoint, "POST", "_search", search_body)
            hits = search_resp["hits"]["hits"]

            if not hits:
                break

            # Buffer docs
            for hit in hits:
                text = hit["_source"].get(text_field, "")
                if not text:
                    text = " "
                chunk_buffer.append(json.dumps({"doc_id": hit["_id"], "text": text}))

                # Flush chunk to S3 when buffer is full
                if len(chunk_buffer) >= CHUNK_SIZE:
                    _flush_chunk(s3_bucket, s3_prefix, chunk_num, chunk_buffer)
                    total_extracted += len(chunk_buffer)
                    chunk_num += 1
                    chunk_buffer = []

                    elapsed = time.time() - start_time
                    rate = total_extracted / elapsed if elapsed > 0 else 0
                    logger.info(f"Extracted {total_extracted:,} docs, {rate:.0f} docs/sec")

            search_after = hits[-1]["sort"]

        # Flush remaining
        if chunk_buffer:
            _flush_chunk(s3_bucket, s3_prefix, chunk_num, chunk_buffer)
            total_extracted += len(chunk_buffer)
            chunk_num += 1

    finally:
        # Always clean up PIT
        try:
            aoss_request(endpoint, "DELETE", "_search/point_in_time", {"pit_id": pit_id})
            logger.info("PIT deleted")
        except Exception as e:
            logger.warning(f"Failed to delete PIT: {e}")

    elapsed = time.time() - start_time
    logger.info(f"ExtractTexts complete: {total_extracted:,} docs in {chunk_num} chunks, "
                f"{elapsed:.0f}s ({total_extracted/max(elapsed,1):.0f} docs/sec)")

    # Write summary for Step Function to read
    summary = {
        "total_extracted": total_extracted,
        "chunks": chunk_num,
        "elapsed_seconds": int(elapsed),
    }
    s3.put_object(
        Bucket=s3_bucket,
        Key=f"{s3_prefix}/extract_summary.json",
        Body=json.dumps(summary).encode("utf-8"),
    )
    logger.info(f"Summary written to s3://{s3_bucket}/{s3_prefix}/extract_summary.json")

    if total_extracted == 0:
        logger.warning("No documents extracted — index may have no docs with text field")


def _cleanup_s3_prefix(s3_bucket, prefix):
    """Delete all objects under an S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        if not objects:
            continue
        s3.delete_objects(
            Bucket=s3_bucket,
            Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
        )
        logger.info(f"Deleted {len(objects)} objects from s3://{s3_bucket}/{prefix}")


def _flush_chunk(s3_bucket, s3_prefix, chunk_num, lines):
    key = f"{s3_prefix}/texts/chunk_{chunk_num:06d}.jsonl"
    body = ("\n".join(lines) + "\n").encode("utf-8")
    if is_encryption_enabled():
        # Prepend nonce to ciphertext (SageMaker input channel loses S3 metadata)
        encrypt_and_upload_with_nonce_prefix(s3, s3_bucket, key, body)
    else:
        s3.put_object(Bucket=s3_bucket, Key=key, Body=body)


if __name__ == "__main__":
    main()
