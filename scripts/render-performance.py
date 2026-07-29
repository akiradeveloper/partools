#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def source_digest() -> str:
    digest = hashlib.sha256()
    paths = [ROOT / "Cargo.toml", ROOT / "massively/Cargo.toml"]
    paths += sorted((ROOT / "massively/src").rglob("*.rs"))
    paths += sorted((ROOT / "massively/benches").rglob("*.rs"))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def command_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def estimate_ns(path: Path) -> float:
    estimates = json.loads(path.read_text())
    estimate = estimates.get("slope") or estimates["mean"]
    return estimate["point_estimate"]


def benchmark_results(baseline):
    root = ROOT / "target" / "criterion" / "performance"
    paths = sorted(root.glob(f"*/{baseline}/estimates.json"))
    if not paths:
        raise SystemExit(f"no Criterion results found under {root.relative_to(ROOT)}")
    source = (ROOT / "massively/benches/performance.rs").read_text()
    expected = set(re.findall(r'benchmark(?:_synchronous)?!\(\s*group,\s*(?:exec,\s*)?"([^"]+)"', source))
    actual = {path.parent.parent.name for path in paths}
    if actual != expected:
        raise SystemExit(f"incomplete baseline: missing {sorted(expected - actual)}, unexpected {sorted(actual - expected)}")
    return [(path.parent.parent.name, estimate_ns(path)) for path in paths]


def decimal(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def duration(nanoseconds: float) -> str:
    if nanoseconds < 1_000:
        return f"{decimal(nanoseconds)} ns"
    if nanoseconds < 1_000_000:
        return f"{decimal(nanoseconds / 1_000)} us"
    if nanoseconds < 1_000_000_000:
        return f"{decimal(nanoseconds / 1_000_000)} ms"
    return f"{decimal(nanoseconds / 1_000_000_000)} s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a complete, named Criterion performance run.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="measure all APIs in a fresh baseline and render it")
    mode.add_argument("--baseline", help="render an existing complete named baseline")
    args = parser.parse_args()
    if args.run:
        measured = datetime.datetime.now(datetime.timezone.utc)
        baseline = measured.strftime("performance-%Y%m%dT%H%M%SZ")
        revision = command_output("git", "rev-parse", "--short", "HEAD")
        if command_output("git", "status", "--short"):
            revision += "-dirty"
        metadata = dict(adapter=os.environ.get("MASSIVELY_BENCH_DEVICE", "wgpu default adapter"),
                        revision=revision, rust=command_output("rustc", "--version"),
                        measured=measured.isoformat(), baseline=baseline, source_sha256=source_digest())
        subprocess.run(["cargo", "bench", "-p", "massively", "--bench", "performance", "--",
                        "--save-baseline", baseline, "--warm-up-time", "1", "--measurement-time", "1",
                        "--sample-size", "20"], cwd=ROOT, check=True)
        metadata_path = ROOT / "target/criterion/performance" / f"{baseline}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        baseline = args.baseline
        if not re.fullmatch(r"[A-Za-z0-9_-]+", baseline):
            parser.error("baseline must be a simple Criterion baseline name")
        metadata_path = ROOT / "target/criterion/performance" / f"{baseline}.json"
        if not metadata_path.exists():
            parser.error(f"measurement provenance is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
    rows = [(api, duration(estimate)) for api, estimate in benchmark_results(baseline)]

    lines = [
        "# Performance",
        "",
        "Reference execution times for the vector API at N = 10,000,000.",
        "The table lists APIs covered by the dedicated performance benchmark.",
        "These are approximate, machine-specific values rather than performance guarantees.",
        "",
        "| Environment | Value |",
        "|---|---|",
        f"| GPU | {metadata['adapter']} |",
        "| Runtime | CubeCL WGPU default device |",
        f"| Rust | `{metadata['rust']}` |",
        f"| Revision | `{metadata['revision']}` |",
        f"| Library and benchmark source SHA-256 | `{metadata.get('source_sha256', 'not recorded')}` |",
        f"| Measured (UTC) | {metadata['measured']} |",
        f"| Criterion baseline | `{baseline}` |",
        "",
        "| API | Time |",
        "|---|---:|",
    ]
    lines.extend(f"| `{api}` | {time} |" for api, time in rows)
    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "Stored inputs are already device-resident, and input construction and host-to-device transfer are "
            "excluded. API-internal output allocation and synchronization required to observe completion are "
            "included; caller-provided output buffers are allocated before timing. Times are Criterion point "
            "estimates after warm-up (slope when available, otherwise mean).",
            "",
            "N is the length of each input range, so binary range algorithms process two inputs of N elements. The "
            "default element type is `f32` for value algorithms and `u32` for ordering, index, and key algorithms. "
            "Predicate and extremum queries use `lazy::counting`, whose item type is `MIndex`. Each "
            "`lower_bound` and `upper_bound` measurement performs N batched searches over an N-element source. "
            "Selection uses 50% nonzero `u32` flags directly as an `MFlag` stencil, indexed "
            "operations use reverse indices, by-key algorithms use runs of eight equal keys, `scatter_reduce` "
            "maps four inputs to each output, and sorting uses deterministic bounded keys "
            "`(i * 1103515245 + 12345) % N` (duplicates are possible).",
            "",
            "Regenerate this file with:",
            "",
            "```console",
            'MASSIVELY_BENCH_DEVICE="<GPU model>" just performance',
            "```",
            "",
        ]
    )
    (ROOT / "PERFORMANCE.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
