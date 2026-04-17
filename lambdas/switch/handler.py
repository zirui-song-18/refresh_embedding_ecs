"""Phase 3b: Switch ingest + search pipelines to v2-only after completeness check passes."""

import logging

from aoss_client import aoss_request

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    endpoint = event["collection_endpoint"]
    index_name = event["index_name"]
    v2_field = event["v2_field"]
    text_field = event["text_field"]
    pipeline_name = event["pipeline_name"]
    search_pipeline_name = event["search_pipeline_name"]
    model_id = event["model_id"]

    # 1. Update ingest pipeline: generate v2 only (stop generating v1)
    logger.info(f"Switching ingest pipeline {pipeline_name} to v2-only ({v2_field})")
    aoss_request(endpoint, "PUT", f"_ingest/pipeline/{pipeline_name}", {
        "description": f"Fine-tuned model pipeline: {text_field} -> {v2_field}",
        "processors": [
            {
                "text_embedding": {
                    "model_id": model_id,
                    "field_map": {text_field: v2_field},
                }
            },
        ],
    })
    logger.info(f"Ingest pipeline switched to v2-only.")

    # 2. Update search pipeline: query on v2 field with fine-tuned model
    logger.info(f"Updating search pipeline {search_pipeline_name} to use {v2_field}")
    aoss_request(endpoint, "PUT", f"_search/pipeline/{search_pipeline_name}", {
        "description": f"Search pipeline: neural query on {v2_field}",
        "request_processors": [
            {
                "neural_query_enricher": {
                    "default_model_id": model_id,
                    "neural_field_default_id": {
                        v2_field: model_id,
                    },
                }
            },
        ],
    })
    logger.info(f"Search pipeline switched to {v2_field}.")

    # 3. Bind search pipeline to index (if not already bound)
    logger.info(f"Binding search pipeline to index {index_name}")
    aoss_request(endpoint, "PUT", f"{index_name}/_settings", {
        "index.search.default_pipeline": search_pipeline_name,
    })
    logger.info(f"Search pipeline bound to index.")

    return {
        "status": "switched",
        "active_field": v2_field,
        "pipeline_name": pipeline_name,
        "search_pipeline_name": search_pipeline_name,
        "index_name": index_name,
    }
