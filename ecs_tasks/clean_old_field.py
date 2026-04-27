"""ECS Task: Set old embedding field to null on all documents to reclaim storage.

Streams per-chunk (same pattern as WriteEmbeddings):
  PIT search → accumulate doc_ids → when chunk full → parallel write → free → continue
Peak memory = one chunk of batch items (~350 bytes × 50K = ~17MB).
"""

import json
import logging
import os
import time

from aoss_client import aoss_request
from bulk_writer import parallel_bulk_write

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEARCH_PAGE_SIZE = 10000
BULK_BATCH_SIZE = 1000
CHUNK_SIZE = 50000  # flush and write after collecting this many doc_ids


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
                "sort": [{"_doc": "asc"}],
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
                        "source": f"ctx._source.{old_field} = null",
                        "lang": "painless",
                    }
                })
                current_chunk.append({"action": action, "body": body})

                # Flush chunk when full
                if len(current_chunk) >= CHUNK_SIZE:
                    s, f = _flush_chunk(endpoint, current_chunk, concurrency, chunk_num)
                    total_success += s
                    total_failures += f
                    chunk_num += 1
                    current_chunk = []

                    elapsed = time.time() - start_time
                    rate = total_success / elapsed if elapsed > 0 else 0
                    logger.info(f"Cleaned {total_success:,}/{total_to_clean:,} docs, "
                                f"{rate:.0f} docs/sec, failures={total_failures}")

            search_after = hits[-1]["sort"]

        # Flush remaining
        if current_chunk:
            s, f = _flush_chunk(endpoint, current_chunk, concurrency, chunk_num)
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


def _flush_chunk(endpoint, chunk_items, concurrency, chunk_num):
    """Split chunk items into batches and parallel write."""
    batches = []
    for i in range(0, len(chunk_items), BULK_BATCH_SIZE):
        batches.append(chunk_items[i:i + BULK_BATCH_SIZE])

    return parallel_bulk_write(
        endpoint, batches, concurrency,
        task_label=f"CleanOldField[chunk {chunk_num}]"
    )


if __name__ == "__main__":
    main()
