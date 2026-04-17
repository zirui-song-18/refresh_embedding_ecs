# Dual-Field Reindex POC v2 — ECS + Lambda Hybrid

Option B architecture: ECS for heavy tasks, Lambda for light tasks, Step Function for orchestration.

## Architecture

```
Step Function orchestrates:
  1. Prepare              → Lambda (~5 sec)
  2. ExtractTexts         → ECS Fargate Task (runs to completion, 30-60 min)
  3. StartEmbeddingJob    → Lambda (~2 sec)
  4. WaitForEmbeddingJob  → Lambda poll loop (60s intervals)
  5. WriteEmbeddings      → ECS Fargate Task (runs to completion, 1-6 hours)
  6. CompletenessCheck    → Lambda (~2 sec)
  7. Switch               → Lambda (~2 sec)
```

## Deploy

```bash
# 1. Build and push Docker image for ECS tasks
docker build -t reindex-ecs-tasks .
aws ecr create-repository --repository-name reindex-ecs-tasks --region us-west-2
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-west-2.amazonaws.com
docker tag reindex-ecs-tasks:latest ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/reindex-ecs-tasks:latest
docker push ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/reindex-ecs-tasks:latest

# 2. Build and deploy SAM stack
sam build
sam deploy --guided \
    --parameter-overrides \
    "ECSImageUri=ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/reindex-ecs-tasks:latest" \
    "SubnetIds=subnet-xxx,subnet-yyy" \
    --capabilities CAPABILITY_NAMED_IAM

# 3. Add ECS task role to AOSS data access policy

# 4. Trigger
python tests/trigger_reindex.py \
    --endpoint $AOSS_ENDPOINT \
    --index test-reindex \
    --model-id $MODEL_ID \
    --s3-bucket reindex-ecs-data-ACCOUNT-REGION \
    --sagemaker-role $SM_ROLE_ARN \
    --state-machine-arn $STATE_MACHINE_ARN \
    --dimension 768 \
    --embedding-model-name Thenlper/gte-base
```

## Key Differences from POC v1 (all-Lambda)

| Aspect | v1 (Lambda) | v2 (ECS + Lambda) |
|--------|-------------|-------------------|
| ExtractTexts | Lambda loop (880 iterations) | Single ECS task (runs to completion) |
| WriteEmbeddings | Lambda loop (880 iterations) | Single ECS task (runs to completion) |
| History events | ~14,000 (57% of limit) | ~750 (3% of limit) |
| Time limit | 15 min per Lambda | No limit (Fargate) |
| Lambda concurrency | 1 per iteration | 0 during heavy tasks |
```
