# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import logging
import os
import toml
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Conveniences to other module directories via relative paths
UWLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
UWLAB_ASSETS_DATA_DIR = os.path.join(UWLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""
UWLAB_ASSETS_METADATA = toml.load(os.path.join(UWLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

UWLAB_CLOUD_ASSETS_DIR = "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main"
DEFAULT_CUSTOM_CLOUD_ASSETS_DIR = "https://huggingface.co/datasets/bowenli1024/ReSYNC_IsaacLab/resolve/main"
CUSTOM_CLOUD_ASSETS_DIR = (
    os.environ.get("CUSTOM_CLOUD_ASSETS_DIR")
    or os.environ.get("UWLAB_CUSTOM_CLOUD_ASSETS_DIR")
    or DEFAULT_CUSTOM_CLOUD_ASSETS_DIR
)
"""Optional project-specific cloud asset root.

Override ``CUSTOM_CLOUD_ASSETS_DIR`` with a public HTTP(S), Omniverse, or local
root that mirrors the relative layout expected by :func:`custom_cloud_path`.
"""


def _join_asset_path(root: str, relative_path: str) -> str:
    """Join an asset root and relative path for URLs, Omniverse paths, or files."""
    if root.startswith(("http://", "https://", "omniverse://")):
        return f"{root.rstrip('/')}/{relative_path.lstrip('/')}"
    return os.path.join(os.path.expanduser(root), relative_path)


def custom_cloud_path(relative_path: str, local_fallback: str) -> str:
    """Return a custom-cloud asset path when configured, otherwise a local path."""
    if CUSTOM_CLOUD_ASSETS_DIR:
        return _join_asset_path(CUSTOM_CLOUD_ASSETS_DIR, relative_path)
    return local_fallback


def _extract_relative_path(url: str) -> str:
    """Strip the HuggingFace resolve-URL prefix, returning the repo-relative path.

    Example:
        ``https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Props/Custom/Peg/peg.usd``
        -> ``Props/Custom/Peg/peg.usd``
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    try:
        idx = parts.index("resolve")
        return "/".join(parts[idx + 2 :])
    except ValueError:
        return parsed.path.strip("/")


def _urlretrieve_quiet(url: str, dest: str) -> None:
    """Download *url* to *dest* silently."""
    req = urllib.request.urlopen(url)
    chunk_size = 1 << 16  # 64 KiB
    with open(dest, "wb") as f:
        while True:
            chunk = req.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    req.close()


def resolve_cloud_path(path: str) -> str:
    """Resolve a cloud asset path to a local file, downloading if needed.

    * Local paths (including already-cached files) are returned immediately.
    * HTTPS URLs are downloaded once to ``~/.cache/uwlab/assets/<relative>``
      and the local cached path is returned on subsequent calls.
    * Downloads are atomic (write to a temp file, then ``os.rename``).
    """
    if not path.startswith(("http://", "https://")):
        return path

    rel = _extract_relative_path(path)
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "uwlab", "assets")
    local = os.path.join(cache_dir, rel)

    if os.path.isfile(local):
        return local

    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = f"{local}.tmp.{os.getpid()}"
    try:
        logger.info(f"Downloading {rel} ...")
        _urlretrieve_quiet(path, tmp)
        os.rename(tmp, local)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    return local


# Configure the module-level variables
__version__ = UWLAB_ASSETS_METADATA["package"]["version"]
