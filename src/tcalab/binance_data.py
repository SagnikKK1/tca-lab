"""Minimal checksum-verified downloader for data.binance.vision bulk files.

Vendored subset of quant-alpha-lab's ingestion client — just enough for the
studies (aggTrades, bookTicker archives). Every zip is verified against its
sibling .CHECKSUM (SHA256) at download time; cached files are trusted
(completed archives are immutable upstream).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

BINANCE_VISION_BASE = "https://data.binance.vision"
UM_FUTURES_PREFIX = "data/futures/um"
RAW_DIR = Path("data/raw")


def has_header(csv_bytes: bytes) -> bool:
    """Binance um files carry a header row only from 2022-01; detect per
    file (first field numeric => headerless)."""
    first_field = csv_bytes.split(b"\n", 1)[0].split(b",", 1)[0].strip()
    try:
        float(first_field)
        return False
    except ValueError:
        return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, session: requests.Session, retries: int = 3) -> bytes | None:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=120)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code >= 500 and attempt < retries - 1:
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        return resp.content
    return None


def download_verified(
    url: str,
    dest: Path | None = None,
    session: requests.Session | None = None,
) -> Path | None:
    session = session or requests.Session()
    if dest is None:
        dest = RAW_DIR / url.removeprefix(BINANCE_VISION_BASE + "/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    checksum_body = _fetch(url + ".CHECKSUM", session)
    if checksum_body is None:
        return None
    expected = checksum_body.decode().split()[0].strip().lower()
    body = _fetch(url, session)
    if body is None:
        return None
    dest.write_bytes(body)
    actual = _sha256(dest)
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {url}")
    return dest
