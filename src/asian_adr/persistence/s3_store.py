"""Object store for the Parquet archive and backtest results.

An :class:`ObjectStore` protocol with a filesystem implementation
(:class:`LocalObjectStore`, the MinIO/local-dev path, fully functional) and an
:class:`S3Store` backend via ``boto3`` (lazy import). Handles raw bytes and
pandas DataFrames (as Parquet).

Imports only the standard library; pandas is used only inside the Parquet
helpers (imported there, so the module loads without it).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = ["ObjectStore", "LocalObjectStore", "S3Store"]


@runtime_checkable
class ObjectStore(Protocol):
    async def put_bytes(self, key: str, data: bytes) -> None: ...
    async def get_bytes(self, key: str) -> bytes | None: ...
    async def exists(self, key: str) -> bool: ...
    async def list_keys(self, prefix: str = "") -> list[str]: ...
    async def put_parquet(self, key: str, df) -> None: ...
    async def get_parquet(self, key: str): ...


def _read_parquet_bytes(data: bytes):
    import pandas as pd

    buf = io.BytesIO(data)
    try:
        return pd.read_parquet(buf)
    except OSError:  # pragma: no cover - engine fallback
        buf.seek(0)
        return pd.read_parquet(buf, engine="fastparquet")


def _write_parquet_bytes(df) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class LocalObjectStore:
    """Filesystem-backed object store rooted at a directory (MinIO/dev path)."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        return self._root / key

    async def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def get_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.exists() else None

    async def exists(self, key: str) -> bool:
        return self._path(key).exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        base = self._root
        if not base.exists():
            return []
        keys = [
            str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()
        ]
        return sorted(k for k in keys if k.startswith(prefix))

    async def put_parquet(self, key: str, df) -> None:
        await self.put_bytes(key, _write_parquet_bytes(df))

    async def get_parquet(self, key: str):
        data = await self.get_bytes(key)
        return _read_parquet_bytes(data) if data is not None else None


class S3Store:
    """S3 / MinIO backend via ``boto3`` (lazy import)."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        prefix: str = "",
    ) -> None:
        self._bucket = bucket
        self._endpoint = endpoint_url
        self._prefix = prefix
        self._client = None

    def _ensure_client(self):  # pragma: no cover - requires boto3
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("S3Store requires 'boto3' (uv add boto3).") from exc
            self._client = boto3.client("s3", endpoint_url=self._endpoint)
        return self._client

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def put_bytes(self, key: str, data: bytes) -> None:  # pragma: no cover
        self._ensure_client().put_object(
            Bucket=self._bucket, Key=self._full_key(key), Body=data
        )

    async def get_bytes(self, key: str) -> bytes | None:  # pragma: no cover
        client = self._ensure_client()
        try:
            resp = client.get_object(Bucket=self._bucket, Key=self._full_key(key))
            return resp["Body"].read()
        except client.exceptions.NoSuchKey:
            return None

    async def exists(self, key: str) -> bool:  # pragma: no cover
        client = self._ensure_client()
        try:
            client.head_object(Bucket=self._bucket, Key=self._full_key(key))
            return True
        except Exception:
            return False

    async def list_keys(self, prefix: str = "") -> list[str]:  # pragma: no cover
        client = self._ensure_client()
        resp = client.list_objects_v2(
            Bucket=self._bucket, Prefix=self._full_key(prefix)
        )
        return [obj["Key"] for obj in resp.get("Contents", [])]

    async def put_parquet(self, key: str, df) -> None:  # pragma: no cover
        await self.put_bytes(key, _write_parquet_bytes(df))

    async def get_parquet(self, key: str):  # pragma: no cover
        data = await self.get_bytes(key)
        return _read_parquet_bytes(data) if data is not None else None
