"""Completeness check: supports both v2 embedding check and old field cleanup check.

Modes (determined by 'check_mode' input):
  - "v2_exists" (default): count docs missing v2_field → complete when 0 missing
  - "clean_complete": count docs still having check_field → complete when 0 remaining
"""

import logging

from aoss_client import aoss_request

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    endpoint = event["collection_endpoint"]
    index_name = event["index_name"]
    check_mode = event.get("check_mode", "v2_exists")

    if check_mode == "clean_complete":
        return _check_clean_complete(event, endpoint, index_name)
    else:
        return _check_v2_exists(event, endpoint, index_name)


def _check_v2_exists(event, endpoint, index_name):
    """Original mode: check if all docs have v2 embedding field."""
    v2_field = event["v2_field"]

    missing_resp = aoss_request(endpoint, "POST", f"{index_name}/_count", {
        "query": {"bool": {"must_not": {"exists": {"field": v2_field}}}}
    })
    missing_count = missing_resp["count"]

    total_resp = aoss_request(endpoint, "POST", f"{index_name}/_count", {
        "query": {"match_all": {}}
    })
    total_count = total_resp["count"]

    completion_pct = round(
        (total_count - missing_count) / max(total_count, 1) * 100, 2
    )

    logger.info(
        f"Completeness: {total_count - missing_count}/{total_count} "
        f"({completion_pct}%) — {missing_count} docs still missing {v2_field}"
    )

    return {
        "missing_count": missing_count,
        "total_count": total_count,
        "complete": missing_count == 0,
        "completion_pct": completion_pct,
        # Pass through config for Switch Lambda
        "collection_endpoint": endpoint,
        "index_name": index_name,
        "v2_field": v2_field,
        "text_field": event["text_field"],
        "pipeline_name": event["pipeline_name"],
        "search_pipeline_name": event["search_pipeline_name"],
        "model_id": event["model_id"],
        "v1_field": event["v1_field"],
    }


def _check_clean_complete(event, endpoint, index_name):
    """Clean mode: check if old field has been nulled on all docs."""
    check_field = event["check_field"]

    remaining_resp = aoss_request(endpoint, "POST", f"{index_name}/_count", {
        "query": {"exists": {"field": check_field}}
    })
    remaining_count = remaining_resp["count"]

    total_resp = aoss_request(endpoint, "POST", f"{index_name}/_count", {
        "query": {"match_all": {}}
    })
    total_count = total_resp["count"]

    cleaned_pct = round(
        (total_count - remaining_count) / max(total_count, 1) * 100, 2
    )

    logger.info(
        f"Clean check: {remaining_count} docs still have {check_field} "
        f"({cleaned_pct}% cleaned)"
    )

    return {
        "remaining_count": remaining_count,
        "total_count": total_count,
        "complete": remaining_count == 0,
        "cleaned_pct": cleaned_pct,
        # Pass through for retry loop
        "collection_endpoint": endpoint,
        "index_name": index_name,
        "check_field": check_field,
    }
