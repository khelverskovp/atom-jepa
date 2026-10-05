#!/usr/bin/env python3
"""Stream-filter Alexandria entries, optionally taking a global reservoir sample."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import bz2
import fcntl
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys
import time

DEFAULT_RELEASE = "2025.07.02"
INPUT_RE = re.compile(r"(alexandria_\d+)\.json(?:\.bz2|\.gz)?")


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


def open_input(path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


class JSONStream:
    """Buffer individual JSON values, accepting the publisher's NaN/Infinity.

    Python's decoder understands those nonstandard numeric values; strict
    streaming parsers reject real Alexandria records with NaN stresses.
    The 128 Mi-character ceiling applies to one value, never the entries array,
    and prevents malformed input from buffering an entire multi-GB shard.
    """
    MAX_VALUE_CHARS = 128 * 1024 * 1024

    def __init__(self, handle, chunk_size=64 * 1024):
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def refill(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        if len(self.buffer) > self.MAX_VALUE_CHARS:
            raise ValueError("Malformed or oversized individual JSON value (>128 Mi characters)")
        block = self.handle.read(self.chunk_size)
        self.buffer += block
        self.eof = not block

    def peek(self):
        while True:
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                if char not in " \t\r\n":
                    return char
                self.position += 1
            if self.eof:
                return ""
            self.refill()

    def expect(self, char):
        found = self.peek()
        if found != char:
            raise ValueError(f"Expected {char!r}, found {found!r} in JSON stream")
        self.position += 1

    def value(self):
        first = self.peek()
        if not first:
            raise ValueError("Unexpected end of JSON document")
        if first not in '{["':
            # Wait for the entire scalar token, including split exponents or
            # NaN/Infinity literals. raw_decode alone can accept a number prefix.
            while True:
                end = re.search(r"[ \t\r\n,\]}:]", self.buffer[self.position:])
                if end or self.eof:
                    stop = self.position + end.start() if end else len(self.buffer)
                    result = json.loads(self.buffer[self.position:stop])
                    self.position = stop
                    return result
                self.refill()
        while True:
            try:
                result, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.refill()
            else:
                self.position = end
                return result

    def array_items(self):
        self.expect("[")
        if self.peek() == "]":
            self.position += 1
            return
        while True:
            yield self.value()
            if self.peek() == "]":
                self.position += 1
                return
            self.expect(",")


def iter_entries(path):
    with open_input(path) as binary:
        with io.TextIOWrapper(binary, encoding="utf-8") as handle:
            stream = JSONStream(handle)
            first = stream.peek()
            if first == "[":
                yield from stream.array_items()
            elif first == "{":
                stream.expect("{")
                found_entries = False
                if stream.peek() != "}":
                    while True:
                        key = stream.value()
                        if not isinstance(key, str):
                            raise ValueError(f"{path}: JSON object key must be a string")
                        stream.expect(":")
                        if key == "entries":
                            if found_entries:
                                raise ValueError(f"{path}: duplicate entries array")
                            found_entries = True
                            yield from stream.array_items()
                        else:
                            stream.value()
                        if stream.peek() == "}":
                            break
                        stream.expect(",")
                stream.expect("}")
                if not found_entries:
                    raise ValueError(f"{path}: no top-level entries array")
            else:
                raise ValueError(f"{path}: expected an entries object or a JSON array")
            # Read through EOF to detect trailing content and truncated compression.
            if stream.peek():
                raise ValueError(f"{path}: trailing data after JSON document")


def selection_reason(entry, *, min_atoms, max_atoms, hull_max, hull_key):
    if not isinstance(entry, dict):
        raise ValueError("entry is not an object")
    structure = entry.get("structure")
    if not isinstance(structure, dict) or not isinstance(structure.get("sites"), list):
        raise ValueError("missing structure.sites array")
    n_atoms = len(structure["sites"])
    if not min_atoms <= n_atoms <= max_atoms:
        return "rejected_atoms"
    value = entry
    try:
        for key in hull_key.split("."):
            value = value[key]
    except (KeyError, TypeError):
        raise ValueError(f"missing {hull_key}")
    if value is None or isinstance(value, bool):
        raise ValueError(f"invalid {hull_key}: {value!r}")
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"non-numeric {hull_key}")
    if not math.isfinite(value):
        raise ValueError(f"non-finite {hull_key}")
    return "kept" if value <= hull_max else "rejected_hull"


def eligible_entries(path, config, counts):
    last_report = time.monotonic()
    for index, entry in enumerate(iter_entries(path)):
        counts["read"] += 1
        try:
            reason = selection_reason(
                entry, min_atoms=config["min_atoms"], max_atoms=config["max_atoms"],
                hull_max=config["e_above_hull_max"], hull_key=config["hull_key"],
            )
        except ValueError as error:
            if not config["skip_invalid"]:
                raise ValueError(f"{path.name}, entry {index}: {error}")
            reason = "invalid"
        counts[reason] += 1
        if time.monotonic() - last_report >= 15:
            print(f"  {path.name}: read {counts['read']:,}; eligible {counts['kept']:,}", flush=True)
            last_report = time.monotonic()
        if reason == "kept":
            yield entry


@contextmanager
def entry_writer(path):
    """Write deterministic gzip; expose a completed entries document only on success."""
    partial = path.with_name(path.name + ".part")
    with partial.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=4) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as handle:
                handle.write('{"entries":[\n')
                first = True

                def write(serialized):
                    nonlocal first
                    if not first:
                        handle.write(",\n")
                    handle.write(serialized)
                    first = False

                yield write
                handle.write("\n]}\n")
    partial.replace(path)


def serialize(entry):
    # Preserve non-finite auxiliary values as in the official source. Hull
    # values are separately required to be finite by selection_reason().
    return json.dumps(entry, ensure_ascii=False, separators=(",", ":"), allow_nan=True)


def validate_download_manifest(directory, paths, allow_partial):
    manifest_path = directory / "download_manifest.json"
    if not manifest_path.exists():
        return {}, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") and not allow_partial:
        raise ValueError("Download is incomplete. Finish downloading, or use "
                         "--allow-partial-input for an intentional smoke test.")
    expected_names = set(manifest.get("available_files", []))
    actual_names = {path.name for path in paths}
    if not allow_partial and actual_names != expected_names:
        raise ValueError("Input files do not match the downloaded release manifest")
    records = {record["name"]: record for record in manifest.get("files", [])}
    if any(path.name not in records for path in paths):
        raise ValueError("Some input shards lack completed download records; rerun the downloader")
    return records, manifest


def source_record(path, download_records):
    record = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    known = download_records.get(path.name)
    if known and any(record[key] != known[key] for key in ("bytes", "sha256")):
        raise ValueError(f"{path.name}: checksum/size differs from download manifest")
    return record


def filter_shard(path, source, output, metadata_dir, config):
    name = INPUT_RE.fullmatch(path.name)[1] + ".json.gz"
    destination = output / name
    done_path = metadata_dir / (name + ".done.json")
    if done_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done["config"] != config or done["source"] != source:
            raise ValueError(f"{name}: source/settings changed; choose another --output-dir")
        if (destination.is_file()
                and destination.stat().st_size == done["output"]["bytes"]
                and sha256_file(destination) == done["output"]["sha256"]):
            print(f"[verified] {name}: {done['counts'].get('kept', 0):,} entries", flush=True)
            return done
    counts = Counter()
    with entry_writer(destination) as write:
        for entry in eligible_entries(path, config, counts):
            write(serialize(entry))
    done = {
        "config": config, "source": source, "counts": dict(counts),
        "output": {"name": name, "bytes": destination.stat().st_size,
                   "sha256": sha256_file(destination)},
    }
    atomic_json(done_path, done)
    print(f"[filtered] {path.name}: {counts['kept']:,}/{counts['read']:,} kept", flush=True)
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--input-dir", type=Path, help="Default: ROOT/raw/RELEASE")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: ROOT/filtered/RELEASE, or ROOT/probes/RELEASE_nN")
    parser.add_argument("--e-above-hull-max", type=float, default=0.05)
    parser.add_argument("--min-atoms", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument("--hull-key", default="data.e_above_hull",
                        help="Dot-separated field, in eV/atom.")
    parser.add_argument("--sample-size", type=int,
                        help="Optional uniform sample AFTER filtering; scans every shard.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-invalid", action="store_true",
                        help="Count and exclude malformed records instead of failing.")
    parser.add_argument("--allow-partial-input", action="store_true",
                        help="Allow an intentionally incomplete download for smoke tests.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", args.release):
        parser.error("--release must have the form YYYY.MM.DD")
    if args.min_atoms < 1 or args.max_atoms < args.min_atoms:
        parser.error("require 1 <= min-atoms <= max-atoms")
    if not math.isfinite(args.e_above_hull_max):
        parser.error("--e-above-hull-max must be finite")
    if not args.hull_key or any(not key for key in args.hull_key.split(".")):
        parser.error("--hull-key must be a nonempty dot-separated path")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be positive")
    directory = (args.input_dir or args.root / "raw" / args.release).resolve()
    output_root = (args.output_dir or (
        args.root / "probes" / f"{args.release}_n{args.sample_size}"
        if args.sample_size else args.root / "filtered" / args.release
    )).resolve()
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    if output_root == directory or directory.is_relative_to(output_root):
        raise ValueError("Choose an output directory separate from the source data")
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and INPUT_RE.fullmatch(path.name))
    if not paths:
        raise ValueError(f"No alexandria_NUMBER.json(.bz2/.gz) shards in {directory}")
    stems = [INPUT_RE.fullmatch(path.name)[1] for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate shard stems with different compression formats")
    download_records, download_manifest = validate_download_manifest(
        directory, paths, args.allow_partial_input)
    if download_manifest and download_manifest["release"] != args.release:
        raise ValueError("--release does not match the download manifest")
    config = {
        "format_version": 2, "min_atoms": args.min_atoms, "max_atoms": args.max_atoms,
        "e_above_hull_max": args.e_above_hull_max, "hull_key": args.hull_key,
        "skip_invalid": args.skip_invalid, "sample_size": args.sample_size,
        "seed": args.seed if args.sample_size else None,
        "atom_count": "len(structure.sites); stored cell; no standardization",
    }
    identity = {"config": config, "inputs": [str(path) for path in paths],
                "release": args.release}
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".filter.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another filter is using this output directory")
        manifest_path = output_root / "filter_manifest.json"
        previous = {}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("identity") != identity:
                raise ValueError("Inputs/settings changed; choose a new --output-dir")
        elif (output_root / "entries").exists() and any((output_root / "entries").iterdir()):
            raise ValueError("Output entries already exist without a manifest; choose a new directory")
        previous_sources = {item["source"]["path"]: item["source"]
                            for item in previous.get("files", [])}
        output = output_root / "entries"
        metadata_dir = output_root / "metadata"
        output.mkdir(exist_ok=True)
        metadata_dir.mkdir(exist_ok=True)
        manifest = {
            "identity": identity, "complete": False, "files": [], "totals": {},
            "parser": "Python json.JSONDecoder; NaN/Infinity auxiliary values preserved",
            "python_version": sys.version.split()[0],
            "download_manifest_sha256": (
                sha256_file(directory / "download_manifest.json") if download_manifest else None),
            "input_release_complete": download_manifest.get("complete") if download_manifest else None,
        }
        atomic_json(manifest_path, manifest)
        print(f"Reading {len(paths)} shards; writing to {output}", flush=True)
        print(f"Filters: e_above_hull <= {args.e_above_hull_max} eV/atom; "
              f"{args.min_atoms} <= stored sites <= {args.max_atoms}", flush=True)
        totals = Counter()
        reservoir = []
        rng = random.Random(args.seed)
        eligible_count = 0
        for path in paths:
            source = source_record(path, download_records)
            if str(path) in previous_sources and previous_sources[str(path)] != source:
                raise ValueError(f"{path.name}: input changed; choose a new --output-dir")
            if args.sample_size is None:
                done = filter_shard(path, source, output, metadata_dir, config)
            else:
                counts = Counter()
                for entry in eligible_entries(path, config, counts):
                    eligible_count += 1
                    slot = (len(reservoir) if len(reservoir) < args.sample_size
                            else rng.randrange(eligible_count))
                    if slot < args.sample_size:
                        item = (eligible_count, serialize(entry))
                        if len(reservoir) < args.sample_size:
                            reservoir.append(item)
                        else:
                            reservoir[slot] = item
                done = {"source": source, "counts": dict(counts)}
                print(f"[scanned] {path.name}: {counts['kept']:,}/{counts['read']:,} eligible", flush=True)
            totals.update(done["counts"])
            manifest["files"].append(done)
            manifest["totals"] = dict(totals)
            atomic_json(manifest_path, manifest)
        if args.sample_size is not None:
            if eligible_count < args.sample_size:
                raise ValueError(f"Only {eligible_count:,} eligible entries; requested "
                                 f"{args.sample_size:,}. Reduce --sample-size.")
            sample_path = output / "alexandria_00000.json.gz"
            with entry_writer(sample_path) as write:
                for _, serialized in sorted(reservoir):
                    write(serialized)
            manifest["sample"] = {
                "name": sample_path.name, "size": len(reservoir),
                "bytes": sample_path.stat().st_size, "sha256": sha256_file(sample_path),
                "method": "Algorithm R reservoir sampling over all eligible entries",
            }
        manifest["written"] = args.sample_size if args.sample_size else totals["kept"]
        manifest["complete"] = True
        atomic_json(manifest_path, manifest)
        print(f"Read {totals['read']:,}; eligible {totals['kept']:,}; "
              f"written {manifest['written']:,}.", flush=True)
        print(f"Rejected by atom count: {totals['rejected_atoms']:,}; "
              f"by hull: {totals['rejected_hull']:,}; invalid: {totals['invalid']:,}.", flush=True)
        print(f"Dataset directory: {output}", flush=True)
        print(f"Manifest: {manifest_path}", flush=True)
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Full filtering resumes by shard; reservoir sampling restarts.",
              file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
