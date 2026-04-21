"""Phase 2b: Start a SageMaker Training Job to generate embeddings in batch.

Packages the training script, uploads to S3, and starts the job.
Training script writes embeddings directly to S3 (bypasses model.tar.gz compression).
"""

import io
import json
import logging
import os
import tarfile
import time

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker = boto3.client("sagemaker")
s3 = boto3.client("s3")

# Configurable via environment variables
INSTANCE_TYPE = os.environ.get("EMBEDDING_INSTANCE_TYPE", "ml.g5.xlarge")
INSTANCE_COUNT = int(os.environ.get("EMBEDDING_INSTANCE_COUNT", "8"))
VOLUME_SIZE_GB = int(os.environ.get("EMBEDDING_VOLUME_SIZE_GB", "50"))

# HuggingFace Training container — has transformers + torch pre-installed
HF_TRAINING_IMAGE = os.environ.get(
    "HF_TRAINING_IMAGE",
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04"
)

# Training script: writes embeddings directly to S3 (no model.tar.gz)
TRAINING_SCRIPT = '''
import json
import logging
import os
import time
from pathlib import Path

import boto3
import torch
from transformers import AutoTokenizer, AutoModel
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --- Encryption helpers ---
def _get_encryption_key():
    key_str = os.environ.get("SM_HP_ENCRYPTION_KEY", "")
    if not key_str:
        return None
    key_bytes = key_str.encode("utf-8")
    if len(key_bytes) < 32:
        key_bytes = key_bytes.ljust(32, b"\\0")
    return key_bytes[:32]


def decrypt_file(file_path, encryption_key):
    """Decrypt an AES-GCM encrypted file. Nonce is first 12 bytes."""
    with open(file_path, "rb") as f:
        data = f.read()
    # For SageMaker: files are downloaded from S3 via input channel.
    # S3 metadata (nonce) is lost, so we prepend nonce to ciphertext during upload.
    nonce = data[:12]
    ciphertext = data[12:]
    aesgcm = AESGCM(encryption_key)
    return aesgcm.decrypt(nonce, ciphertext, None)


def encrypt_and_upload(s3_client, bucket, key, plaintext, encryption_key):
    """Encrypt with AES-GCM and upload. Nonce stored in S3 metadata."""
    nonce = os.urandom(12)
    aesgcm = AESGCM(encryption_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    s3_client.put_object(
        Bucket=bucket, Key=key, Body=ciphertext,
        Metadata={"encryption-nonce": nonce.hex()},
    )
    return key


# --- Model helpers ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_batch(model, tokenizer, texts, max_length, device):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**encoded)
    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def upload_chunk_to_s3(s3_client, bucket, prefix, host_idx, chunk_num, lines, encryption_key):
    """Upload embedding chunk, with optional encryption."""
    key = f"{prefix}/embeddings/host{host_idx:02d}_chunk_{chunk_num:06d}.jsonl"
    body = ("\\n".join(lines) + "\\n").encode("utf-8")
    if encryption_key:
        encrypt_and_upload(s3_client, bucket, key, body, encryption_key)
    else:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body)
    return key


def main():
    input_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_name = os.environ.get("SM_HP_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(os.environ.get("SM_HP_BATCH_SIZE", "256"))
    max_seq_length = int(os.environ.get("SM_HP_MAX_SEQ_LENGTH", "512"))
    chunk_size = int(os.environ.get("SM_HP_CHUNK_SIZE", "50000"))
    s3_bucket = os.environ.get("SM_HP_S3_BUCKET", "")
    s3_prefix = os.environ.get("SM_HP_S3_PREFIX", "")

    # Multi-instance: get host index for unique output naming
    current_host = os.environ.get("SM_CURRENT_HOST", "algo-1")
    host_idx = int(current_host.split("-")[1]) - 1
    num_hosts = len(os.environ.get("SM_HOSTS", "[\\"algo-1\\"]").strip("[]").split(","))
    logger.info(f"Host: {current_host} (index={host_idx}), total hosts: {num_hosts}")

    # Encryption key (None = no encryption)
    encryption_key = _get_encryption_key()
    if encryption_key:
        logger.info("Client-side encryption ENABLED")
    else:
        logger.info("Client-side encryption DISABLED")

    model_channel = Path(os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model"))
    if model_channel.exists() and any(model_channel.iterdir()):
        logger.info(f"Loading fine-tuned model from {model_channel}")
        model_name = str(model_channel)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Model: {model_name}, batch_size: {batch_size}, device: {device}")
    logger.info(f"S3 output: s3://{s3_bucket}/{s3_prefix}/embeddings/")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    s3_client = boto3.client("s3")

    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        logger.warning("No input files found")
        (model_dir / "done.marker").write_text("no input")
        return

    logger.info(f"Found {len(input_files)} input files (ShardedByS3Key assigned to this host)")

    total_written = 0
    chunk_num = 0
    chunk_ids = []
    chunk_texts = []
    start_time = time.time()

    for input_file in input_files:
        logger.info(f"Reading {input_file}")

        # Decrypt if encryption enabled
        if encryption_key:
            plaintext = decrypt_file(input_file, encryption_key)
            file_lines = plaintext.decode("utf-8").splitlines()
        else:
            with open(input_file, "r", encoding="utf-8") as in_f:
                file_lines = in_f.readlines()

        for line in file_lines:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chunk_ids.append(record["doc_id"])
            chunk_texts.append(record.get("text", ""))

            if len(chunk_texts) >= chunk_size:
                emb_lines = _encode_chunk(model, tokenizer, chunk_ids, chunk_texts,
                                          batch_size, max_seq_length, device)
                key = upload_chunk_to_s3(s3_client, s3_bucket, s3_prefix, host_idx,
                                         chunk_num, emb_lines, encryption_key)

                total_written += len(chunk_texts)
                elapsed = time.time() - start_time
                rate = total_written / elapsed if elapsed > 0 else 0
                logger.info(f"Chunk {chunk_num}: {len(chunk_texts)} docs -> s3://{s3_bucket}/{key} | "
                            f"Total: {total_written} | {rate:.0f} docs/sec")

                chunk_num += 1
                chunk_ids = []
                chunk_texts = []

    # Final partial chunk
    if chunk_texts:
        emb_lines = _encode_chunk(model, tokenizer, chunk_ids, chunk_texts,
                                  batch_size, max_seq_length, device)
        key = upload_chunk_to_s3(s3_client, s3_bucket, s3_prefix, host_idx,
                                 chunk_num, emb_lines, encryption_key)
        total_written += len(chunk_texts)
        chunk_num += 1
        logger.info(f"Final chunk: {len(chunk_texts)} docs -> s3://{s3_bucket}/{key}")

    elapsed = time.time() - start_time
    logger.info(f"Done. Host {current_host}: {total_written} embeddings in {chunk_num} chunks, "
                f"{elapsed:.0f}s ({total_written/max(elapsed,1):.0f} docs/sec)")

    (model_dir / "done.marker").write_text(
        json.dumps({"host": current_host, "total_written": total_written,
                    "chunks": chunk_num, "elapsed_seconds": int(elapsed)})
    )


def _encode_chunk(model, tokenizer, doc_ids, texts, batch_size, max_length, device):
    lines = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = doc_ids[i:i + batch_size]
        embeddings = encode_batch(model, tokenizer, batch_texts, max_length, device)
        for doc_id, emb in zip(batch_ids, embeddings):
            lines.append(json.dumps({"doc_id": doc_id, "embedding": emb.tolist()}))
    return lines


if __name__ == "__main__":
    main()
'''


def _upload_training_script(s3_bucket, s3_prefix):
    """Package training script as sourcedir.tar.gz and upload to S3."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        script_bytes = TRAINING_SCRIPT.encode("utf-8")
        info = tarfile.TarInfo(name="generate_embeddings.py")
        info.size = len(script_bytes)
        tar.addfile(info, io.BytesIO(script_bytes))
    buf.seek(0)

    s3_key = f"{s3_prefix}/scripts/sourcedir.tar.gz"
    s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=buf.read())
    logger.info(f"Uploaded training script to s3://{s3_bucket}/{s3_key}")
    return f"s3://{s3_bucket}/{s3_key}"


def handler(event, context):
    s3_bucket = event["s3_bucket"]
    s3_prefix = event["s3_prefix"]
    sagemaker_role = event["sagemaker_role"]
    model_name = event.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = event.get("embedding_batch_size", 256)

    # Upload training script to S3
    script_s3_uri = _upload_training_script(s3_bucket, s3_prefix)

    # S3 paths
    input_s3_uri = f"s3://{s3_bucket}/{s3_prefix}/texts/"
    output_s3_uri = f"s3://{s3_bucket}/{s3_prefix}/output/"

    # Unique job name
    job_name = f"embed-{s3_prefix.replace('/', '-')[:40]}-{int(time.time())}"
    job_name = "".join(c if c.isalnum() or c == "-" else "-" for c in job_name)[:63]

    logger.info(f"Starting Training Job: {job_name}")
    logger.info(f"Input: {input_s3_uri}")
    logger.info(f"Output: {output_s3_uri}")
    logger.info(f"Instance: {INSTANCE_TYPE}")

    sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": HF_TRAINING_IMAGE,
            "TrainingInputMode": "File",
        },
        RoleArn=sagemaker_role,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_s3_uri,
                        "S3DataDistributionType": "ShardedByS3Key",
                    }
                },
                "ContentType": "application/jsonlines",
            }
        ],
        OutputDataConfig={
            "S3OutputPath": output_s3_uri,
        },
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": INSTANCE_COUNT,
            "VolumeSizeInGB": VOLUME_SIZE_GB,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 86400,  # 1 day
        },
        HyperParameters={
            "sagemaker_program": "generate_embeddings.py",
            "sagemaker_submit_directory": script_s3_uri,
            "model_name": model_name,
            "batch_size": str(batch_size),
            "chunk_size": str(event.get("chunk_size", 50000)),
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
            "encryption_key": os.environ.get("ENCRYPTION_KEY", ""),
        },
    )

    logger.info(f"Training Job {job_name} started")

    return {
        "job_name": job_name,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "output_s3_uri": output_s3_uri,
        # Pass through config
        "collection_endpoint": event["collection_endpoint"],
        "index_name": event["index_name"],
        "v2_field": event["v2_field"],
        "text_field": event["text_field"],
        "pipeline_name": event["pipeline_name"],
        "search_pipeline_name": event["search_pipeline_name"],
        "model_id": event["model_id"],
        "v1_field": event["v1_field"],
    }
