"""Phase 2c: Poll SageMaker Training Job status."""

import logging
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker = boto3.client("sagemaker")


def handler(event, context):
    job_name = event["job_name"]

    resp = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = resp["TrainingJobStatus"]
    # Possible: InProgress, Completed, Failed, Stopping, Stopped

    logger.info(f"Training Job {job_name}: {status}")

    result = {
        "job_name": job_name,
        "status": status,
        "completed": status == "Completed",
        "failed": status in ("Failed", "Stopped"),
    }

    if status == "Completed":
        # The output is at: {OutputDataConfig.S3OutputPath}/{job_name}/output/model.tar.gz
        output_s3 = resp.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
        result["model_artifacts_s3"] = output_s3
        logger.info(f"Output: {output_s3}")

    if status == "Failed":
        failure_reason = resp.get("FailureReason", "Unknown")
        result["failure_reason"] = failure_reason
        logger.error(f"Job failed: {failure_reason}")

    # Pass through all config
    for key in ["s3_bucket", "s3_prefix", "output_s3_uri",
                 "collection_endpoint", "index_name", "v2_field", "text_field",
                 "pipeline_name", "search_pipeline_name", "model_id", "v1_field"]:
        if key in event:
            result[key] = event[key]

    return result
