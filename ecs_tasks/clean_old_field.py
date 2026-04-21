"""ECS Task: Set old embedding field to null on all documents to reclaim storage.

Runs after Switch — the old field is no longer used for queries or ingestion.
Uses PIT + search_after to paginate, bulk update to null.
"""

import json
import logging
import os
import time

import boto3
from aoss_client import aoss_request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEARCH_PAGE_SIZE = 10000
BULK_BATCH_SIZE = 1000  # larger than WriteEmbeddings since payload is tiny


def main():
    endpoint = os.environ["COLLECTION_ENDPOINT"]
    index_name = os.environ["INDEX_NAME"]
    old_field = os.environ["OLD_FIELD"]

    logger.info(f"CleanOldField starting: index={index_name}, old_field={old_field}")

    # Count docs that still have the old field (non-null)
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

    total_cleaned = 0
    total_failures = 0
    search_after = None
    start_time = time.time()
    backoff_delay = 0.0

    try:
        while True:
            # Search for docs that have the old field
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

            # Build bulk request to null out the old field
            doc_ids = [hit["_id"] for hit in hits]

            for batch_start in range(0, len(doc_ids), BULK_BATCH_SIZE):
                batch_ids = doc_ids[batch_start:batch_start + BULK_BATCH_SIZE]

                if backoff_delay > 0:
                    time.sleep(backoff_delay)

                bulk_lines = []
                for doc_id in batch_ids:
                    action = json.dumps({"update": {"_id": doc_id, "_index": index_name}})
                    script = json.dumps({
                        "script": {
                            "source": f"ctx._source.{old_field} = null",
                            "lang": "painless",
                        }
                    })
                    bulk_lines.append(action)
                    bulk_lines.append(script)

                bulk_body = "\n".join(bulk_lines) + "\n"

                try:
                    bulk_resp = aoss_request(endpoint, "POST", "_bulk", bulk_body)
                except RuntimeError as e:
                    total_failures += len(batch_ids)
                    backoff_delay = min(backoff_delay + 5, 60)
                    logger.warning(f"Bulk failed ({len(batch_ids)} docs), backoff={backoff_delay}s: {e}")
                    continue

                # Count results
                batch_failures = 0
                for item in bulk_resp.get("items", []):
                    result = item.get("update", {})
                    status = result.get("status", 0)
                    if status == 200:
                        total_cleaned += 1
                    elif status == 429:
                        batch_failures += 1
                    else:
                        total_failures += 1

                if batch_failures > 0:
                    total_failures += batch_failures
                    backoff_delay = min(backoff_delay + 2, 60)
                else:
                    backoff_delay = max(backoff_delay - 1, 0)

            search_after = hits[-1]["sort"]

            elapsed = time.time() - start_time
            rate = total_cleaned / elapsed if elapsed > 0 else 0
            logger.info(f"Cleaned {total_cleaned:,}/{total_to_clean:,} docs, "
                        f"{rate:.0f} docs/sec, failures={total_failures}")

    finally:
        try:
            aoss_request(endpoint, "DELETE", "_search/point_in_time", {"pit_id": pit_id})
            logger.info("PIT deleted")
        except Exception as e:
            logger.warning(f"Failed to delete PIT: {e}")

    elapsed = time.time() - start_time
    logger.info(f"CleanOldField complete: cleaned={total_cleaned:,}, "
                f"failures={total_failures:,}, time={elapsed:.0f}s "
                f"({total_cleaned / max(elapsed, 1):.0f} docs/sec)")


if __name__ == "__main__":
    main()
