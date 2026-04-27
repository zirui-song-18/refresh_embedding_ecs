"""AOSS client with SigV4 signing, timeout, retry, and credential refresh.

Thread-safe: uses per-thread HTTP sessions to avoid request corruption
when called from multiple threads via bulk_writer.parallel_bulk_write.
"""

import json
import logging
import threading

import boto3
import requests
from requests.adapters import HTTPAdapter
from requests_aws4auth import AWS4Auth
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_session = boto3.Session()
_retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST", "PUT", "DELETE"],
    raise_on_status=False,
)
_thread_local = threading.local()


def _get_http_session():
    """Get or create a per-thread HTTP session."""
    if not hasattr(_thread_local, "http_session"):
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=_retry_strategy))
        _thread_local.http_session = session
    return _thread_local.http_session


def _get_auth():
    creds = _session.get_credentials().get_frozen_credentials()
    return AWS4Auth(
        creds.access_key, creds.secret_key,
        _session.region_name or "us-west-2", "aoss",
        session_token=creds.token,
    )


def aoss_request(endpoint, method, path, body=None):
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Content-Type": "application/json"}
    if isinstance(body, str):
        headers["Content-Type"] = "application/x-ndjson"
        data = body
    elif body is not None:
        data = json.dumps(body)
    else:
        data = None

    http_session = _get_http_session()
    resp = http_session.request(
        method, url, auth=_get_auth(), headers=headers, data=data,
        timeout=(5, 120),
    )
    if resp.status_code >= 400:
        logger.error(f"AOSS {method} {path} -> {resp.status_code}: {resp.text[:1000]}")
        raise RuntimeError(f"AOSS {method} {path} -> {resp.status_code}: {resp.text[:500]}")
    if resp.text:
        return resp.json()
    return {}
