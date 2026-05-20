# Dual-Field Reindex POC — ECS + Lambda Hybrid

ECS for heavy tasks, Lambda for light tasks, Step Function for orchestration.
Per-chunk 100% guarantee eliminates the need for an outer completeness retry loop.

## Architecture

```
Step Function orchestrates:
  1. Prepare              → Lambda (~5 sec): add v2 mapping, update pipeline
  2. ExtractTexts         → ECS Fargate (extracts all docs with text field)
  3. StartEmbeddingJob    → Lambda: starts SageMaker Training Job
  4. CheckEmbeddingJob    → Lambda poll loop (60s intervals)
  5. WriteEmbeddings      → ECS Fargate (per-chunk 100% guarantee)
  6. Switch               → Lambda: update ingest + search pipelines to v2
  7. CleanOldField         → ECS Fargate (per-page 100% guarantee)
```

No CompletenessCheck loop — WriteEmbeddings guarantees every chunk is fully
written before moving to the next. If the task crashes, the Painless script's
idempotent check (`if field == null → write; else noop`) handles retry safely.

## Prerequisites

- AWS account with AOSS collection, ECR repository, SageMaker endpoint
- [Finch](https://github.com/runfinch/finch) installed (container runtime)
- SAM CLI installed
- Python 3.10+ with `boto3`, `requests`, `requests_aws4auth`

## Deploy

```bash
cd ~/re_index_poc_ecs

# 1. Build and push Docker image for ECS tasks (using Finch)
aws ecr get-login-password --region us-west-2 | \
    finch login --username AWS --password-stdin 330700426359.dkr.ecr.us-west-2.amazonaws.com

finch build --platform linux/amd64 -t reindex-ecs-tasks .
finch tag reindex-ecs-tasks:latest 330700426359.dkr.ecr.us-west-2.amazonaws.com/reindex-ecs-tasks:latest
finch push 330700426359.dkr.ecr.us-west-2.amazonaws.com/reindex-ecs-tasks:latest

# 2. Build and deploy SAM stack (Lambdas + Step Function)
sam build
sam deploy
```

## Test (Small Scale — 200 docs)

### Step 1: Setup test index

```bash
python tests/setup_test_index.py \
    --endpoint https://5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com \
    --index test-ecs-v11 \
    --model-id 702cbcf5-a188-4177-a187-7fb7e63cdbec \
    --sagemaker-endpoint hf-textembedding-all-minilm-l6-v2-2026-04-13-03-04-52-445 \
    --num-docs 200 \
    --dimension 384
```

### Step 2: Trigger reindex via Step Function

```bash
python tests/trigger_reindex.py \
    --endpoint https://5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com \
    --index test-ecs-v11 \
    --model-id 702cbcf5-a188-4177-a187-7fb7e63cdbec \
    --s3-bucket reindex-ecs-data-330700426359-us-west-2 \
    --sagemaker-role arn:aws:iam::330700426359:role/reindex-sagemaker-execution-role-ecs \
    --state-machine-arn arn:aws:states:us-west-2:330700426359:stateMachine:DualFieldReindexECS \
    --dimension 384 \
    --embedding-model-name sentence-transformers/all-MiniLM-L6-v2
```

This will poll and print status until the Step Function completes.

### Step 3: Verify results

```bash
python tests/verify_results.py \
    --endpoint https://5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com \
    --index test-ecs-v11 \
    --v2-field embedding_v2 \
    --v1-field search_text
```

## Test (Large Scale — 8.8M docs, ms-marco)

```bash
python tests/trigger_reindex.py \
    --endpoint https://5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com \
    --index ms-marco \
    --model-id 702cbcf5-a188-4177-a187-7fb7e63cdbec \
    --s3-bucket reindex-ecs-data-330700426359-us-west-2 \
    --sagemaker-role arn:aws:iam::330700426359:role/reindex-sagemaker-execution-role-ecs \
    --state-machine-arn arn:aws:states:us-west-2:330700426359:stateMachine:DualFieldReindexECS \
    --v1-field embedding_v8 \
    --v2-field embedding_v9 \
    --text-field text \
    --dimension 768 \
    --embedding-model-name Thenlper/gte-base \
    --poll-interval 60

# Verify
python tests/verify_results.py \
    --endpoint https://5l2oxy87av1hbwupnpoc.us-west-2.aoss.amazonaws.com \
    --index ms-marco \
    --v2-field embedding_v9 \
    --v1-field embedding_v8
```

Expected runtime: ~3-5 hours (Extract ~30min, SageMaker ~1-2h, Write ~1-2h, Clean ~30min).

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Extract ALL docs (not just missing v2) | Simpler; Painless noop handles already-written docs on retry |
| Per-chunk 100% guarantee | Eliminates outer retry loop; no wasted re-extraction or re-embedding |
| Sort `[_doc, _id]` | `_id` tiebreaker ensures no missed docs across AOSS shards |
| Painless uses `params.field` + `[]` | Prevents script injection via field name |
| Adaptive backoff with jitter | Prevents thundering herd when concurrent threads wake |
| `bulk_writer` returns `failed_items` | Enables precise retry of only failed docs (not entire batch) |
