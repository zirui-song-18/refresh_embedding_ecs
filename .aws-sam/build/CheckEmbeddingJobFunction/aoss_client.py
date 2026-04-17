"""Shared AOSS client — SigV4-signed HTTP requests with timeout and retry."""

import json
import logging
import boto3
import requests
from requests.adapters import HTTPAdapter
from requests_aws4auth import AWS4Auth
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_session = boto3.Session()

# Retry adapter: retries on 429 (throttle), 500, 502, 503
_retry_strategy = Retry(
    total=3,
    backoff_factor=1,          # wait 1s, 2s, 4s between retries
    status_forcelist=[429, 500, 502, 503],
    allowed_methods=["GET", "POST", "PUT", "DELETE"],
    raise_on_status=False,     # don't raise, let us handle the response
)
_http_session = requests.Session()
_http_session.mount("https://", HTTPAdapter(max_retries=_retry_strategy))


def _get_auth():
    """Get fresh SigV4 auth credentials (handles Lambda container reuse)."""
    creds = _session.get_credentials().get_frozen_credentials()
    return AWS4Auth(
        creds.access_key,
        creds.secret_key,
        _session.region_name or "us-west-2",
        "aoss",
        session_token=creds.token,
    )


def aoss_request(endpoint: str, method: str, path: str, body=None):
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"

    headers = {"Content-Type": "application/json"}
    if isinstance(body, str):
        headers["Content-Type"] = "application/x-ndjson"
        data = body
    elif body is not None:
        data = json.dumps(body)
    else:
        data = None

    resp = _http_session.request(
        method, url,
        auth=_get_auth(),
        headers=headers,
        data=data,
        timeout=(5, 60),      # 5s connect, 60s read
    )

    if resp.status_code >= 400:
        logger.error(f"AOSS {method} {path} -> {resp.status_code}: {resp.text[:1000]}")
        raise RuntimeError(f"AOSS {method} {path} -> {resp.status_code}: {resp.text[:500]}")

    if resp.text:
        return resp.json()
    return {}
