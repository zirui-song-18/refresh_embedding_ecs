"""ECS Task: Set old embedding field to null on all documents to reclaim storage.

Per-page 100% guarantee: retries failed docs within each page until all
succeed before advancing search_after. No outer clean retry loop needed.

Streams per-chunk (same pattern as WriteEmbeddings):
  PIT search → accumulate doc_ids → when chunk full → parallel write with
  page-level retry → free → continue
"""

import json
import logging
import os
import time

from aoss_client import aoss_request
from backoff import AdaptiveBackoff
from bulk_writer import parallel_bulk_write

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEARCH_PAGE_SIZE = 10000
BULK_BATCH_SIZE = 1000
CHUNK_SIZE = 50000  # flush and write after collecting this many doc_ids
_MAX_PAGE_RETRIES = 10


def main():
    endpoint = os.environ["COLLECTION_ENDPOINT"]
    index_name = os.environ["INDEX_NAME"]
    old_field = os.environ["OLD_FIELD"]
    concurrency = int(os.environ.get("WRITE_CONCURRENCY", "1"))

    logger.info(f"CleanOldField starting: index={index_name}, old_field={old_field}, "
                f"concurrency={concurrency}")

    count_resp = aoss_request(endpoint, "POST", f"{index_name}/_count", {
        "query": {"exists": {"field": old_field}}
    })
    total_to_clean = count_resp.get("count", 0)
    logger.info(f"Documents with {old_field}: {total_to_clean:,}")

    if total_to_clean == 0:
        logger.info("No documents to clean — old field already null on all docs")
        return

    # Create PIT
    pit_resp = aoss_request(
        endpoint, "POST",
        f"{index_name}/_search/point_in_time?keep_alive=15m",
    )
    pit_id = pit_resp["pit_id"]
    logger.info("PIT created")

    total_success = 0
    total_failures = 0
    chunk_num = 0
    current_chunk = []  # accumulates batch items until CHUNK_SIZE
    search_after = None
    start_time = time.time()

    try:
        while True:
            search_body = {
                "size": SEARCH_PAGE_SIZE,
                "_source": False,
                "query": {"exists": {"field": old_field}},
                "pit": {"id": pit_id, "keep_alive": "15m"},
                "sort": [{"_doc": "asc"}, {"_id": "asc"}],
            }
            if search_after:
                search_body["search_after"] = search_after

            search_resp = aoss_request(endpoint, "POST", "_search", search_body)
            hits = search_resp["hits"]["hits"]

            if not hits:
                break

            # Build batch items from this page
            for hit in hits:
                action = json.dumps({"update": {"_id": hit["_id"], "_index": index_name}})
                body = json.dumps({
                    "script": {
                        "source": "ctx._source[params.field] = null",
                        "lang": "painless",
                        "params": {"field": old_field},
                    }
                })
                current_chunk.append({"action": action, "body": body})

                # Flush chunk when full — with per-chunk 100% guarantee
                if len(current_chunk) >= CHUNK_SIZE:
                    s, f = _flush_chunk_with_retry(
                        endpoint, current_chunk, concurrency, chunk_num
                    )
                    total_success += s
                    total_failures += f
                    chunk_num += 1
                    current_chunk = []

                    elapsed = time.time() - start_time
                    rate = total_success / elapsed if elapsed > 0 else 0
                    logger.info(f"Cleaned {total_success:,}/{total_to_clean:,} docs, "
                                f"{rate:.0f} docs/sec, failures={total_failures}")

            search_after = hits[-1]["sort"]

        # Flush remaining — with per-chunk 100% guarantee
        if current_chunk:
            s, f = _flush_chunk_with_retry(
                endpoint, current_chunk, concurrency, chunk_num
            )
            total_success += s
            total_failures += f

    finally:
        try:
            aoss_request(endpoint, "DELETE", "_search/point_in_time", {"pit_id": pit_id})
            logger.info("PIT deleted")
        except Exception as e:
            logger.warning(f"Failed to delete PIT: {e}")

    elapsed = time.time() - start_time
    logger.info(f"CleanOldField complete: cleaned={total_success:,}, failures={total_failures:,}, "
                f"time={elapsed:.0f}s ({total_success / max(elapsed, 1):.0f} docs/sec)")


def _flush_chunk_with_retry(endpoint, chunk_items, concurrency, chunk_num):
    """Write chunk with per-chunk 100% guarantee. Returns (success, failures)."""
    remaining_items = chunk_items
    total_success = 0
    page_retry = 0
    page_backoff = AdaptiveBackoff(initial=5.0, max_delay=120.0)

    while remaining_items:
        batches = [remaining_items[i:i + BULK_BATCH_SIZE]
                   for i in range(0, len(remaining_items), BULK_BATCH_SIZE)]

        success, failed_items = parallel_bulk_write(
            endpoint, batches, concurrency,
            task_label=f"CleanOldField[chunk {chunk_num}]"
        )
        total_success += success
        remaining_items = failed_items

        if remaining_items:
            page_retry += 1
            if page_retry >= _MAX_PAGE_RETRIES:
                logger.error(
                    f"Chunk {chunk_num}: {len(remaining_items)} docs failed after "
                    f"{_MAX_PAGE_RETRIES} page retries — giving up"
                )
                return total_success, len(remaining_items)
            else:
                logger.warning(
                    f"Chunk {chunk_num}: {len(remaining_items)} docs need page-level retry "
                    f"(round {page_retry}/{_MAX_PAGE_RETRIES})"
                )
                page_backoff.on_failure()
                page_backoff.wait()

    return total_success, 0


if __name__ == "__main__":
    main()
