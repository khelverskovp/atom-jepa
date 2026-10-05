#!/usr/bin/env python3
"""Download a dated Alexandria PBE 3D release; Python 3.10+, standard library only."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

DEFAULT_RELEASE = "2025.07.02"
SHARD_RE = re.compile(r"alexandria_\d+\.json\.bz2")
PRINT_LOCK = threading.Lock()
STOP_EVENT = threading.Event()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(url, *, method="GET", headers=None, timeout=60):
    values = {
        "User-Agent": "AlexandriaDatasetDownloader/1.0",
        "Accept-Encoding": "identity",
    }
    values.update(headers or {})
    return urlopen(Request(url, headers=values, method=method), timeout=timeout)


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def discover_shards(source_url, timeout=60):
    with request(source_url, timeout=timeout) as response:
        html = response.read().decode("utf-8")
    parser = LinkParser()
    parser.feed(html)
    origin = urlsplit(source_url)
    shards = {}
    for href in parser.links:
        url = urljoin(source_url, href)
        parts = urlsplit(url)
        name = unquote(parts.path.rsplit("/", 1)[-1])
        if (parts.scheme, parts.netloc) != (origin.scheme, origin.netloc):
            continue
        if parts.path.rsplit("/", 1)[0] != origin.path.rstrip("/"):
            continue
        if SHARD_RE.fullmatch(name) and not parts.query and not parts.fragment:
            shards[name] = url
    if not shards:
        raise ValueError(f"No Alexandria PBE shards found at {source_url}")
    # Excludes convex_hull.json.bz2 and optimization trajectories.
    return sorted(shards.items())


def remote_metadata(url, timeout):
    with request(url, method="HEAD", timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is None or int(length) <= 0:
            raise ValueError(f"Missing/invalid Content-Length for {url}")
        return {
            "url": url,
            "bytes": int(length),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def download_once(name, url, directory, old_record, timeout):
    if STOP_EVENT.is_set():
        raise concurrent.futures.CancelledError()
    metadata = remote_metadata(url, timeout)
    destination = directory / name
    partial = directory / (name + ".part")
    partial_meta = directory / (name + ".part.meta")

    if old_record and old_record.get("url") == url:
        for key in ("etag", "last_modified", "bytes"):
            old_value, new_value = old_record.get(key), metadata.get(key)
            if old_value is not None and new_value is not None and old_value != new_value:
                raise ValueError(
                    f"{name}: source metadata changed since the recorded download. "
                    "Use a fresh --output-dir to keep release versions separate."
                )
        if (destination.is_file()
                and destination.stat().st_size == metadata["bytes"]
                and sha256_file(destination) == old_record.get("sha256")):
            log(f"[verified] {name}")
            return dict(metadata, name=name, sha256=old_record["sha256"])

    # Resume only with matching source validators.
    saved = None
    if partial_meta.is_file():
        saved = json.loads(partial_meta.read_text(encoding="utf-8"))
    etag = metadata["etag"]
    validator = (etag if etag and not etag.startswith("W/")
                 else metadata["last_modified"])
    offset = partial.stat().st_size if partial.is_file() else 0
    if saved != metadata or offset > metadata["bytes"] or (offset and not validator):
        partial.unlink(missing_ok=True)
        offset = 0
    atomic_json(partial_meta, metadata)

    if offset < metadata["bytes"]:
        headers = {}
        if offset:
            headers = {"Range": f"bytes={offset}-", "If-Range": validator}
        try:
            response = request(url, headers=headers, timeout=timeout)
        except HTTPError as error:
            if error.code == 416:
                partial.unlink(missing_ok=True)
                partial_meta.unlink(missing_ok=True)
            raise
        with response:
            encoding = response.headers.get("Content-Encoding", "identity")
            if encoding != "identity":
                raise ValueError(f"{name}: unexpected HTTP content encoding {encoding}")
            for key, header in (("etag", "ETag"), ("last_modified", "Last-Modified")):
                value = response.headers.get(header)
                if value and metadata[key] and value != metadata[key]:
                    raise ValueError(f"{name}: source changed during the download")
            if response.status == 206:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)",
                    response.headers.get("Content-Range", ""),
                )
                if not match or int(match[1]) != offset or int(match[3]) != metadata["bytes"]:
                    partial.unlink(missing_ok=True)
                    raise ValueError(f"{name}: invalid Content-Range; refusing mixed bytes")
                mode = "ab" if offset else "wb"
            elif response.status == 200:
                # If Range is ignored, restart instead of appending a full response.
                mode, offset = "wb", 0
            else:
                raise ValueError(f"{name}: unexpected HTTP status {response.status}")
            log(f"[download] {name}, starting at {offset / 2**20:.1f} MiB")
            last_report = time.monotonic()
            with partial.open(mode) as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    if STOP_EVENT.is_set():
                        raise concurrent.futures.CancelledError()
                    handle.write(block)
                    offset += len(block)
                    if offset > metadata["bytes"]:
                        raise ValueError(f"{name}: response exceeds expected file size")
                    if time.monotonic() - last_report >= 15:
                        log(f"  {name}: {offset / 2**20:.1f}/{metadata['bytes'] / 2**20:.1f} MiB")
                        last_report = time.monotonic()
    if partial.stat().st_size != metadata["bytes"]:
        raise OSError(f"{name}: incomplete download; retained .part for resuming")
    with partial.open("rb") as handle:
        if handle.read(3) != b"BZh":
            partial.unlink(missing_ok=True)
            raise ValueError(f"{name}: response is not a bzip2 file")
    checksum = sha256_file(partial)
    partial.replace(destination)
    partial_meta.unlink(missing_ok=True)
    log(f"[saved] {name} ({metadata['bytes'] / 2**20:.1f} MiB)")
    return dict(metadata, name=name, sha256=checksum)


def download_with_retries(name, url, directory, old_record, timeout, retries):
    for attempt in range(retries):
        try:
            return download_once(name, url, directory, old_record, timeout)
        except (OSError, URLError) as error:
            if attempt + 1 == retries:
                raise
            delay = min(2 ** attempt, 10)
            log(f"[retry {attempt + 1}/{retries}] {name}: {error}")
            time.sleep(delay)


def main(argv=None):
    STOP_EVENT.clear()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", default=DEFAULT_RELEASE,
                        help="Dated release, e.g. 2025.07.02; use your pretraining release.")
    parser.add_argument("--source-url", help="Override the dated PBE 3D directory URL.")
    parser.add_argument("--output-dir", type=Path, help="Default: ROOT/raw/RELEASE")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-files", type=int,
                        help="Smoke tests only: download the first N shards.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List URLs without downloading or creating directories.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", args.release):
        parser.error("--release must have the form YYYY.MM.DD")
    if min(args.workers, args.retries, args.timeout) <= 0:
        parser.error("workers, retries, and timeout must be positive")
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive")
    source_url = (args.source_url or
                  f"https://alexandria.icams.rub.de/data/pbe/{args.release}/").rstrip("/") + "/"
    directory = (args.output_dir or args.root / "raw" / args.release).resolve()
    all_shards = discover_shards(source_url, args.timeout)
    selected = all_shards[:args.max_files] if args.max_files else all_shards
    log(f"Release {args.release}: {len(all_shards)} shards; selected {len(selected)}.")
    log(f"Output: {directory}")
    if args.dry_run:
        for _, url in selected:
            print(url)
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another downloader is using this output directory")
        manifest_path = directory / "download_manifest.json"
        previous = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists() else {})
        if previous and previous.get("source_url") != source_url:
            raise ValueError("Output directory belongs to another source; use --output-dir")
        old_records = {item["name"]: item for item in previous.get("files", [])}
        # Retain completed hashes while a rerun verifies files, even if interrupted.
        names = {name for name, _ in all_shards}
        records = {name: value for name, value in old_records.items() if name in names}
        manifest = {
            "format_version": 1, "release": args.release, "source_url": source_url,
            "available_files": [name for name, _ in all_shards],
            "selected_files": [name for name, _ in selected],
            "complete": False, "files": list(records.values()), "errors": [],
        }
        atomic_json(manifest_path, manifest)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download_with_retries, name, url, directory,
                            old_records.get(name), args.timeout, args.retries): name
                for name, url in selected
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    name = futures[future]
                    try:
                        records[name] = future.result()
                    except Exception as error:
                        manifest["errors"].append({"name": name, "error": str(error)})
                        log(f"[error] {name}: {error}")
                    manifest["files"] = [records[key] for key in sorted(records)]
                    atomic_json(manifest_path, manifest)
            except KeyboardInterrupt:
                STOP_EVENT.set()
                for future in futures:
                    future.cancel()
                raise
        manifest["complete"] = (
            len(selected) == len(all_shards)
            and len(records) == len(all_shards)
            and not manifest["errors"]
        )
        atomic_json(manifest_path, manifest)
        log(f"Verified {len(records)} files, {sum(r['bytes'] for r in records.values()) / 10**9:.2f} GB.")
        if manifest["errors"]:
            return 1
        if not manifest["complete"]:
            log("Partial release: filtering requires --allow-partial-input for this smoke test.")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, URLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
