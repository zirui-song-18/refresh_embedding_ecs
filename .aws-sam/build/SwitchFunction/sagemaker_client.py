"""Shared SageMaker inference client — invoke embedding endpoint with sub-batching."""

import json
import logging
import boto3

logger = logging.getLogger(__name__)

_sm_runtime = boto3.client("sagemaker-runtime")

SM_MAX_BATCH = 256  # TEI container's max_client_batch_size


def get_embeddings(endpoint_name: str, texts: list[str]) -> list[list[float]]:
    """Call a SageMaker embedding endpoint and return vectors.

    Automatically sub-batches if len(texts) > SM_MAX_BATCH.

    Args:
        endpoint_name: SageMaker endpoint name.
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    all_embeddings = []

    for i in range(0, len(texts), SM_MAX_BATCH):
        chunk = texts[i:i + SM_MAX_BATCH]
        embeddings = _invoke_batch(endpoint_name, chunk)
        all_embeddings.extend(embeddings)
        if len(texts) > SM_MAX_BATCH:
            logger.info(f"SageMaker sub-batch {i // SM_MAX_BATCH + 1}: "
                        f"{len(chunk)} texts -> {len(embeddings)} embeddings")

    return all_embeddings


def _invoke_batch(endpoint_name: str, texts: list[str]) -> list[list[float]]:
    """Invoke SageMaker for a single batch (<= SM_MAX_BATCH texts)."""
    payload = json.dumps({"inputs": texts}).encode("utf-8")

    response = _sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode("utf-8"))

    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            return result
        return [result]

    raise ValueError(f"Unexpected SageMaker response format: {type(result)}")
