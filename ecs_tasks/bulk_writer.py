"""Shared parallel bulk writer for AOSS.

Used by both WriteEmbeddings and CleanOldField. Pattern:
  Caller provides batches of bulk request items, this module fans them out
  to a ThreadPoolExecutor with shared AdaptiveBackoff and per-batch 429 retry.

A "batch" is a list of {"action": str, "body": str} dicts representing
the two lines of each bulk request item.

Returns (total_success, failed_items) — callers use failed_items for
chunk/page-level retry.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from aoss_client import aoss_request
from backoff import AdaptiveBackoff

logger = logging.getLogger(__name__)

_MAX_BATCH_RETRIES = 3


def parallel_bulk_write(endpoint, batches, concurrency, task_label="bulk_write"):
    """Write multiple batches to AOSS in parallel.

    Returns:
        (total_success, failed_items) tuple.
        failed_items is a list of batch items that couldn't be written
        (for chunk/page-level retry by caller).
    """
    if not batches:
        return 0, []

    start_time = time.time()
    total_success = 0
    all_failed_items = []
    completed = 0
    backoff = AdaptiveBackoff()  # shared across all threads for global throttle awareness

    if concurrency <= 1:
        for batch in batches:
            s, failed = _write_one_batch(endpoint, batch, backoff)
            total_success += s
            all_failed_items.extend(failed)
            completed += 1
            if completed % 50 == 0 or completed == len(batches):
                _log_progress(task_label, completed, len(batches),
                              total_success, len(all_failed_items), start_time)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_write_one_batch, endpoint, batch, backoff): batch
                for batch in batches
            }
            for future in as_completed(futures):
                try:
                    s, failed = future.result()
                except Exception as e:
                    batch = futures[future]
                    logger.error(f"{task_label} batch failed with exception: {e}")
                    s, failed = 0, batch  # entire batch failed
                total_success += s
                all_failed_items.extend(failed)
                completed += 1

                if completed % 50 == 0 or completed == len(batches):
                    _log_progress(task_label, completed, len(batches),
                                  total_success, len(all_failed_items), start_time)

    elapsed = time.time() - start_time
    logger.info(f"{task_label} complete: success={total_success:,}, "
                f"failures={len(all_failed_items)}, time={elapsed:.0f}s "
                f"({total_success / max(elapsed, 1):.0f} docs/sec)")

    return total_success, all_failed_items


def _write_one_batch(endpoint, batch, backoff):
    """Write a single batch to AOSS with per-batch retry on 429.

    Returns:
        (success_count, failed_items) — failed_items for chunk/page retry.
    """
    success = 0
    failed_items = []
    remaining = batch

    for attempt in range(_MAX_BATCH_RETRIES):
        if not remaining:
            break

        backoff.wait()

        bulk_lines = []
        for item in remaining:
            bulk_lines.append(item["action"])
            bulk_lines.append(item["body"])

        bulk_body = "\n".join(bulk_lines) + "\n"

        try:
            bulk_resp = aoss_request(endpoint, "POST", "_bulk", bulk_body)
        except Exception as e:
            backoff.on_failure()
            if attempt < _MAX_BATCH_RETRIES - 1:
                logger.warning(f"Bulk failed (attempt {attempt + 1}/{_MAX_BATCH_RETRIES}, "
                               f"{len(remaining)} docs), backoff={backoff.delay:.1f}s: {e}")
                continue
            else:
                # All batch retries exhausted → return for chunk/page-level retry
                failed_items.extend(remaining)
                logger.error(f"Bulk permanently failed ({len(remaining)} docs): {e}")
                remaining = []
                break

        resp_items = bulk_resp.get("items", [])

        # Unmatched records → failed
        if len(resp_items) < len(remaining):
            for idx in range(len(resp_items), len(remaining)):
                failed_items.append(remaining[idx])

        throttled = []
        for idx, resp_item in enumerate(resp_items):
            if idx >= len(remaining):
                break
            result = resp_item.get("update", {})
            status = result.get("status", 0)
            if status == 200:
                success += 1
            elif status == 429:
                throttled.append(remaining[idx])
            elif status == 404:
                success += 1  # doc deleted during processing, goal satisfied
            else:
                # Non-429 errors (500, etc.) → chunk/page-level retry
                failed_items.append(remaining[idx])

        remaining = throttled
        if remaining:
            if attempt < _MAX_BATCH_RETRIES - 1:
                backoff.on_failure()
                logger.warning(f"Batch attempt {attempt + 1}: {len(remaining)} throttled, "
                               f"retrying with backoff={backoff.delay:.1f}s")
            else:
                # 429 exhausted batch retries → chunk/page-level retry
                failed_items.extend(remaining)
                logger.warning(f"Batch attempt {attempt + 1}: {len(remaining)} throttled, "
                               f"no batch retries left — deferring to chunk/page retry")
        else:
            backoff.on_success()

    return success, failed_items


def _log_progress(label, completed, total, success, failures, start_time):
    elapsed = time.time() - start_time
    rate = success / elapsed if elapsed > 0 else 0
    logger.info(f"{label}: {completed}/{total} batches, "
                f"success={success:,}, failures={failures}, {rate:.0f} docs/sec")
