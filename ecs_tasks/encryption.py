"""Client-side encryption/decryption for S3 objects using AES-256-GCM.

POC: Uses a fixed key from ENCRYPTION_KEY environment variable.
Production: Key would come from KMS GenerateDataKey / Decrypt.

Usage:
    from encryption import encrypt_and_upload, download_and_decrypt

    # Write encrypted
    encrypt_and_upload(s3, bucket, key, plaintext_bytes)

    # Read encrypted
    plaintext_bytes = download_and_decrypt(s3, bucket, key)
"""

import logging
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# POC: 32-byte key from env var. Production: from KMS.
_ENCRYPTION_KEY: bytes | None = None


def _get_key() -> bytes:
    """Get the encryption key (lazy load from env)."""
    global _ENCRYPTION_KEY
    if _ENCRYPTION_KEY is None:
        key_str = os.environ.get("ENCRYPTION_KEY", "")
        if not key_str:
            raise RuntimeError("ENCRYPTION_KEY environment variable not set")
        # Key must be exactly 32 bytes for AES-256
        key_bytes = key_str.encode("utf-8")
        if len(key_bytes) < 32:
            key_bytes = key_bytes.ljust(32, b"\0")
        _ENCRYPTION_KEY = key_bytes[:32]
    return _ENCRYPTION_KEY


def encrypt_and_upload(s3_client, bucket: str, key: str, plaintext: bytes) -> None:
    """Encrypt data with AES-256-GCM and upload to S3.

    Nonce (12 bytes) is stored in S3 object metadata.
    """
    aesgcm = AESGCM(_get_key())
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=ciphertext,
        Metadata={"encryption-nonce": nonce.hex()},
    )


def encrypt_and_upload_with_nonce_prefix(s3_client, bucket: str, key: str, plaintext: bytes) -> None:
    """Encrypt and upload with nonce PREPENDED to ciphertext (no metadata dependency).

    Used for files that will be downloaded by SageMaker input channel,
    which loses S3 metadata. Format: [12-byte nonce][ciphertext]
    """
    aesgcm = AESGCM(_get_key())
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    # Prepend nonce so the file is self-contained
    s3_client.put_object(Bucket=bucket, Key=key, Body=nonce + ciphertext)


def download_and_decrypt(s3_client, bucket: str, key: str) -> bytes:
    """Download from S3 and decrypt with AES-256-GCM.

    Nonce is read from S3 object metadata.
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    ciphertext = response["Body"].read()

    nonce_hex = response["Metadata"]["encryption-nonce"]
    nonce = bytes.fromhex(nonce_hex)

    aesgcm = AESGCM(_get_key())
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext


# Alias for clarity — SageMaker output uses nonce in S3 metadata
download_and_decrypt_from_metadata = download_and_decrypt


def is_encryption_enabled() -> bool:
    """Check if encryption is enabled (ENCRYPTION_KEY is set)."""
    return bool(os.environ.get("ENCRYPTION_KEY", ""))
