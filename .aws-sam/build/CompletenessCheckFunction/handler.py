"""Phase 3a: Check if all documents have the v2 embedding field."""

import logging

from aoss_client import aoss_request

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    endpoint = event["collection_endpoint"]
    index_name = event["index_name"]
    v2_field = event["v2_field"]

    # Count docs missing v2
    missing_resp = aoss_request(endpoint, "POST", f"{index_name}/_count", {
        "query": {
            "bool": {
                "must_not": {"exists": {"field": v2_field}}
            }
        }
    })
    missing_count = missing_resp["count"]

    # Count total docs
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
