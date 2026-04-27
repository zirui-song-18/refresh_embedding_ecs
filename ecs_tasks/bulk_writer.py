"""Shared parallel bulk writer for AOSS.

Used by both WriteEmbeddings and CleanOldField. Pattern:
  Caller provides batches of bulk request items, this module fans them out
  to a ThreadPoolExecutor with shared AdaptiveBackoff and per-batch 429 retry.

A "batch" is a list of {"action": str, "body": str} dicts representing
the two lines of each bulk request item.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from aoss_client import aoss_request
from backoff import AdaptiveBackoff

logger = logging.getLogger(__name__)


def parallel_bulk_write(endpoint, batches, concurrency, task_label="bulk_write"):
    """Write multiple batches to AOSS in parallel.

    Returns:
        (total_success, total_failures) tuple
    """
    if not batches:
        return 0, 0

    start_time = time.time()
    total_success = 0
    total_failures = 0
    completed = 0
    backoff = AdaptiveBackoff()  # shared across all threads for global throttle awareness

    if concurrency <= 1:
        for batch in batches:
            s, f = _write_one_batch(endpoint, batch, backoff)
            total_success += s
            total_failures += f
            completed += 1
            if completed % 50 == 0 or completed == len(batches):
                _log_progress(task_label, completed, len(batches),
                              total_success, total_failures, start_time)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_write_one_batch, endpoint, batch, backoff): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                try:
                    s, f = future.result()
                except Exception as e:
                    logger.error(f"{task_label} batch failed with exception: {e}")
                    s, f = 0, len(batches[futures[future]])
                total_success += s
                total_failures += f
                completed += 1

                if completed % 50 == 0 or completed == len(batches):
                    _log_progress(task_label, completed, len(batches),
                                  total_success, total_failures, start_time)

    elapsed = time.time() - start_time
    logger.info(f"{task_label} complete: success={total_success:,}, failures={total_failures:,}, "
                f"time={elapsed:.0f}s ({total_success / max(elapsed, 1):.0f} docs/sec)")

    return total_success, total_failures


def _write_one_batch(endpoint, batch, backoff):
    """Write a single batch to AOSS with per-batch retry on 429.

    Returns:
        (success_count, failure_count)
    """
    success = 0
    failures = 0
    remaining = batch

    for attempt in range(3):
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
        except RuntimeError as e:
            backoff.on_failure()
            if attempt < 2:
                logger.warning(f"Bulk failed (attempt {attempt + 1}/3, {len(remaining)} docs), "
                               f"backoff={backoff.delay:.1f}s: {e}")
                continue
            else:
                failures += len(remaining)
                logger.error(f"Bulk permanently failed ({len(remaining)} docs): {e}")
                break

        resp_items = bulk_resp.get("items", [])

        # Count unmatched records as failures
        if len(resp_items) < len(remaining):
            failures += len(remaining) - len(resp_items)

        throttled = []
        for idx, resp_item in enumerate(resp_items):
            result = resp_item.get("update", {})
            status = result.get("status", 0)
            if status == 200:
                success += 1
            elif status == 429 and idx < len(remaining):
                throttled.append(remaining[idx])
            elif status == 404:
                success += 1  # doc deleted during processing, goal satisfied
            else:
                failures += 1

        remaining = throttled
        if remaining:
            backoff.on_failure()
        else:
            backoff.on_success()

    # Any remaining after all attempts
    failures += len(remaining)

    return success, failures


def _log_progress(label, completed, total, success, failures, start_time):
    elapsed = time.time() - start_time
    rate = success / elapsed if elapsed > 0 else 0
    logger.info(f"{label}: {completed}/{total} batches, "
                f"success={success:,}, failures={failures}, {rate:.0f} docs/sec")
