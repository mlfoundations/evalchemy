#!/usr/bin/env python3
"""
Discover and cache HF batch sizes for mv_exp eval jobs.
"""

import argparse
from contextlib import contextmanager
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import fcntl


DISCOVERY_PATTERNS = [
    re.compile(r"HF estimated max batch size to be (\d+)"),
    re.compile(r"Determined Largest batch size: (\d+)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe lm_eval's auto batch size once and cache the discovered numeric value."
    )
    parser.add_argument("--model", required=True, help="Model identifier passed to eval.eval.")
    parser.add_argument("--task", required=True, help="Benchmark task name passed to eval.eval.")
    parser.add_argument("--cache-file", required=True, help="Path to the JSON cache file.")
    parser.add_argument("--output-path", default="logs", help="eval.eval output path.")
    return parser.parse_args()


def _load_cache_locked(handle) -> dict[str, dict[str, int]]:
    handle.seek(0)
    raw = handle.read().strip()
    if not raw:
        return {}

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Batch size cache must contain a top-level JSON object.")
    return parsed


def _get_lock_file(cache_file: Path) -> Path:
    return cache_file.with_name(f"{cache_file.name}.lock")


@contextmanager
def _exclusive_cache_lock(cache_file: Path):
    lock_file = _get_lock_file(cache_file)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_cache_unlocked(cache_file: Path) -> dict[str, dict[str, int]]:
    if not cache_file.exists():
        return {}

    with cache_file.open("r", encoding="utf-8") as handle:
        return _load_cache_locked(handle)


def _atomic_write_cache_unlocked(
    cache_file: Path, cache: dict[str, dict[str, int]]
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(cache, indent=2, sort_keys=True) + "\n"
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=cache_file.parent,
            prefix=f".{cache_file.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)

        os.replace(tmp_path, cache_file)

        try:
            dir_fd = os.open(str(cache_file.parent), os.O_RDONLY)
        except OSError:
            return

        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def read_cached_batch_size(cache_file: Path, model: str, task: str) -> int | None:
    with _exclusive_cache_lock(cache_file):
        cache = _read_cache_unlocked(cache_file)

    batch_size = cache.get(model, {}).get(task)
    return batch_size if isinstance(batch_size, int) and batch_size > 0 else None


def write_cached_batch_size(cache_file: Path, model: str, task: str, batch_size: int) -> None:
    with _exclusive_cache_lock(cache_file):
        cache = _read_cache_unlocked(cache_file)
        model_cache = cache.setdefault(model, {})
        model_cache[task] = batch_size
        _atomic_write_cache_unlocked(cache_file, cache)


def build_probe_command(args: argparse.Namespace) -> list[str]:
    model_args = f"trust_remote_code=True,pretrained={args.model}"
    return [
        "accelerate",
        "launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "-m",
        "eval.eval",
        "--model",
        "hf",
        "--tasks",
        args.task,
        "--model_args",
        model_args,
        "--batch_size",
        "auto",
        "--output_path",
        args.output_path,
    ]


def extract_batch_size(text: str) -> int | None:
    for pattern in DISCOVERY_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def terminate_probe(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=15)


def discover_batch_size(command: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
        env=env,
    )

    output_lines: list[str] = []
    discovered_batch_size: int | None = None

    try:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()

            discovered_batch_size = extract_batch_size(line)
            if discovered_batch_size is not None:
                sys.stderr.write(
                    f"Discovered batch size {discovered_batch_size}; terminating probe before full eval.\n"
                )
                sys.stderr.flush()
                terminate_probe(process)
                break

        if discovered_batch_size is None:
            process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()

    if discovered_batch_size is None:
        full_output = "".join(output_lines)
        discovered_batch_size = extract_batch_size(full_output)

    if discovered_batch_size is None:
        raise RuntimeError("Failed to detect batch size from probe output.")

    return discovered_batch_size


def main() -> int:
    args = parse_args()
    cache_file = Path(args.cache_file)

    cached_batch_size = read_cached_batch_size(cache_file, args.model, args.task)
    if cached_batch_size is not None:
        sys.stderr.write(
            f"Using cached batch size {cached_batch_size} for {args.model} / {args.task}\n"
        )
        print(cached_batch_size)
        return 0

    discovered_batch_size = discover_batch_size(build_probe_command(args))
    write_cached_batch_size(cache_file, args.model, args.task, discovered_batch_size)
    print(discovered_batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
