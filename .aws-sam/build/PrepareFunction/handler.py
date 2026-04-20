"""Phase 1: Add search_text_v2 field + update ingest pipeline to dual-model."""

import json
import logging
import re

from aoss_client import aoss_request

VALID_FIELD_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    """Prepare the index for dual-field reindex.

    1. Add a new knn_vector field (v2) to the index mapping.
    2. Update the ingest pipeline to generate BOTH v1 and v2 embeddings.

    Input event:
        collection_endpoint: str  - AOSS endpoint URL
        index_name: str           - Target index
        v1_field: str             - Existing embedding field (e.g. "search_text")
        v2_field: str             - New embedding field (e.g. "search_text_v2")
        text_field: str           - Source text field (e.g. "text")
        pipeline_name: str        - Ingest pipeline name
        model_id: str             - ML model ID registered in AOSS
        dimension: int            - Embedding dimension (e.g. 384)
    """
    endpoint = event["collection_endpoint"]
    index_name = event["index_name"]
    v1_field = event["v1_field"]
    v2_field = event["v2_field"]
    text_field = event["text_field"]
    pipeline_name = event["pipeline_name"]
    model_id = event["model_id"]
    dimension = event["dimension"]

    # Validate field names to prevent Painless script injection
    for field_name, field_label in [(v1_field, "v1_field"), (v2_field, "v2_field"), (text_field, "text_field")]:
        if not VALID_FIELD_PATTERN.match(field_name):
            raise ValueError(f"Invalid {field_label}: '{field_name}'. Must match pattern {VALID_FIELD_PATTERN.pattern}")

    # Step 1: Add v2 knn_vector field to mapping
    logger.info(f"Adding {v2_field} field (dim={dimension}) to index {index_name}")
    aoss_request(endpoint, "PUT", f"{index_name}/_mapping", {
        "properties": {
            v2_field: {
                "type": "knn_vector",
                "dimension": dimension,
            }
        }
    })
    logger.info(f"Mapping updated: {v2_field} added")

    # Step 2: Update ingest pipeline to generate both v1 and v2
    # POC: uses the same model for both fields (proves mechanism)
    # Production: v1 = base model, v2 = fine-tuned model
    logger.info(f"Updating pipeline {pipeline_name} to dual-model")
    aoss_request(endpoint, "PUT", f"_ingest/pipeline/{pipeline_name}", {
        "description": f"Dual-model pipeline: {text_field} -> {v1_field} AND {v2_field}",
        "processors": [
            {
                "text_embedding": {
                    "model_id": model_id,
                    "field_map": {text_field: v1_field},
                }
            },
            {
                "text_embedding": {
                    "model_id": model_id,
                    "field_map": {text_field: v2_field},
                }
            },
        ],
    })
    logger.info(f"Pipeline {pipeline_name} updated to dual-model")

    # Step 3: Get doc count and determine ECS resource tier
    count_resp = aoss_request(endpoint, "POST", f"{index_name}/_count", {
        "query": {"match_all": {}}
    })
    doc_count = count_resp.get("count", 0)
    logger.info(f"Index {index_name} has {doc_count:,} documents")

    # Tier: small < 1M, medium 1M-50M, large > 50M
    # Fargate valid combos: 2vCPU/4-16GB, 4vCPU/8-30GB, 8vCPU/16-60GB
    if doc_count < 1_000_000:
        ecs_cpu, ecs_memory, chunk_size = "2048", "10240", 50000
        tier = "small"
    elif doc_count < 50_000_000:
        ecs_cpu, ecs_memory, chunk_size = "4096", "30720", 50000
        tier = "medium"
    else:
        ecs_cpu, ecs_memory, chunk_size = "8192", "61440", 10000
        tier = "large"

    logger.info(f"Tier: {tier} — ECS config: {ecs_cpu} CPU, {ecs_memory} MB, chunk_size={chunk_size}")

    return {
        "status": "prepared",
        "index_name": index_name,
        "v2_field": v2_field,
        "pipeline_name": pipeline_name,
        "doc_count": doc_count,
        "ecs_cpu": ecs_cpu,
        "ecs_memory": ecs_memory,
        "chunk_size": chunk_size,
        "tier": tier,
    }
