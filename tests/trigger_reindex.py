#!/usr/bin/env python3
"""Trigger the dual-field reindex Step Function and monitor with detailed timing.

Designed to run on EC2 or local machine. Only requires boto3 (no AOSS dependencies).

Usage:
    python trigger_reindex.py \
        --endpoint https://xxx.us-west-2.aoss.amazonaws.com \
        --index msmarco \
        --model-id YOUR_MODEL_ID \
        --s3-bucket reindex-embedding-data-ACCOUNT-REGION \
        --sagemaker-role arn:aws:iam::ACCOUNT:role/reindex-sagemaker-execution-role \
        --state-machine-arn arn:aws:states:... \
        --dimension 768 \
        --embedding-model-name Thenlper/gte-base
"""

import argparse
import json
import time
import uuid
from datetime import datetime, timezone

import boto3


def fmt_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="AOSS collection endpoint")
    parser.add_argument("--index", default="test-reindex")
    parser.add_argument("--model-id", required=True, help="ML model ID in AOSS")
    parser.add_argument("--pipeline-name", default="test-reindex-pipeline")
    parser.add_argument("--search-pipeline-name", default="test-reindex-search-pipeline")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket for intermediate data")
    parser.add_argument("--sagemaker-role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--state-machine-arn", required=True)
    parser.add_argument("--v1-field", default="search_text")
    parser.add_argument("--v2-field", default="search_text_v2")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--embedding-model-name", default="Thenlper/gte-base",
                        help="HuggingFace model name for embedding generation")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status polls")
    args = parser.parse_args()

    sfn = boto3.client("stepfunctions", region_name="us-west-2")

    task_id = str(uuid.uuid4())[:8]
    s3_prefix = f"reindex/{args.index}/{args.v2_field}/{task_id}"

    input_payload = {
        "collection_endpoint": args.endpoint,
        "index_name": args.index,
        "v1_field": args.v1_field,
        "v2_field": args.v2_field,
        "text_field": args.text_field,
        "pipeline_name": args.pipeline_name,
        "search_pipeline_name": args.search_pipeline_name,
        "model_id": args.model_id,
        "dimension": args.dimension,
        "s3_bucket": args.s3_bucket,
        "s3_prefix": s3_prefix,
        "sagemaker_role": args.sagemaker_role,
        "embedding_model_name": args.embedding_model_name,
    }

    # Print configuration
    print("=" * 70)
    print(f"DUAL-FIELD REINDEX — LOAD TEST")
    print("=" * 70)
    print(f"Start time:          {datetime.now(timezone.utc).isoformat()}")
    print(f"Index:               {args.index}")
    print(f"v1 field:            {args.v1_field}")
    print(f"v2 field:            {args.v2_field}")
    print(f"Embedding model:     {args.embedding_model_name}")
    print(f"Dimension:           {args.dimension}")
    print(f"S3 path:             s3://{args.s3_bucket}/{s3_prefix}/")
    print(f"SageMaker role:      {args.sagemaker_role}")
    print(f"State machine:       {args.state_machine_arn}")
    print("=" * 70)

    resp = sfn.start_execution(
        stateMachineArn=args.state_machine_arn,
        input=json.dumps(input_payload),
    )
    execution_arn = resp["executionArn"]
    print(f"\nExecution ARN: {execution_arn}\n")

    # Track timing per phase
    start_time = time.time()
    phase_times = {}
    current_phase = None
    phase_start = start_time

    while True:
        status_resp = sfn.describe_execution(executionArn=execution_arn)
        status = status_resp["status"]

        if status != "RUNNING":
            break

        # Get current state
        try:
            history = sfn.get_execution_history(
                executionArn=execution_arn,
                maxResults=5,
                reverseOrder=True,
            )
            current_state = "..."
            for evt in history.get("events", []):
                if "stateEnteredEventDetails" in evt:
                    current_state = evt["stateEnteredEventDetails"].get("name", "...")
                    break
        except Exception:
            current_state = "..."

        # Track phase transitions
        phase = _classify_phase(current_state)
        if phase != current_phase:
            now = time.time()
            if current_phase:
                phase_times[current_phase] = phase_times.get(current_phase, 0) + (now - phase_start)
            current_phase = phase
            phase_start = now

        elapsed = time.time() - start_time
        print(f"  [{fmt_duration(elapsed):>8}] {status} — {current_state}")
        time.sleep(args.poll_interval)

    # Final phase timing
    if current_phase:
        phase_times[current_phase] = phase_times.get(current_phase, 0) + (time.time() - phase_start)

    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    if status == "SUCCEEDED":
        output = json.loads(status_resp.get("output", "{}"))
        print(f"REINDEX SUCCEEDED")
        print(f"Output: {json.dumps(output, indent=2)}")
    else:
        print(f"REINDEX {status}")
        output_str = status_resp.get("output", "")
        if output_str:
            try:
                output = json.loads(output_str)
                print(f"Output: {json.dumps(output, indent=2)}")
            except Exception:
                print(f"Output: {output_str[:1000]}")

    print("\n" + "-" * 70)
    print("TIMING BREAKDOWN")
    print("-" * 70)
    for phase, duration in phase_times.items():
        pct = (duration / total_time * 100) if total_time > 0 else 0
        print(f"  {phase:<30} {fmt_duration(duration):>10}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<30} {fmt_duration(total_time):>10}")
    print("-" * 70)

    print(f"\nEnd time: {datetime.now(timezone.utc).isoformat()}")


def _classify_phase(state_name):
    """Classify a Step Function state into a high-level phase."""
    if state_name in ("Prepare",):
        return "Prepare"
    elif state_name in ("ExtractTexts", "InitExtractLoop", "IsExtractionDone", "PassExtractState"):
        return "ExtractTexts"
    elif state_name in ("StartEmbeddingJob",):
        return "StartEmbeddingJob"
    elif state_name in ("WaitForEmbeddingJob", "CheckEmbeddingJob", "IsJobComplete"):
        return "SageMaker Training Job"
    elif state_name in ("WriteEmbeddings", "InitWriteLoop", "IsWriteDone", "PassWriteState"):
        return "WriteEmbeddings"
    elif state_name in ("WaitForRefresh",):
        return "WaitForRefresh"
    elif state_name in ("CompletenessCheck", "IsComplete", "HandleIncomplete"):
        return "CompletenessCheck"
    elif state_name in ("Switch",):
        return "Switch"
    else:
        return state_name


if __name__ == "__main__":
    main()
