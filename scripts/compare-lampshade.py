#!/usr/bin/env python3
"""Build the pinned upstream comparison runners, then measure them sequentially.

All checkouts, build products, samples, and patches stay under this project.
The comparison changes only dependency paths and provenance labels; inputs,
validation and timing stay upstream.
"""

import argparse
from collections import defaultdict
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_URL = "https://github.com/SamJSui/lampshade"
REFERENCE_REVISION = "ad644d34ebbaaf74e32d2afc2c82874859b1b518"
WORKLOADS = ["reduce_sum", "exclusive_scan", "compact_50", "sort_bounded16", "sort_full_width"]


def command(*args, cwd=ROOT, **kwargs):
    return subprocess.run(args, cwd=cwd, check=True, **kwargs)


def capture(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"pinned runner no longer matches expected text: {old!r}")
    return text.replace(old, new, 1)


def source_digest(source):
    digest = hashlib.sha256()
    paths = [source / "Cargo.toml", source / "massively/Cargo.toml"]
    paths.extend(sorted((source / "massively/src").rglob("*.rs")))
    for path in paths:
        digest.update(str(path.relative_to(source)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_reference(cache):
    reference = cache / "reference"
    if not reference.exists():
        command("git", "init", str(reference), stdout=subprocess.DEVNULL)
        command("git", "fetch", "--depth", "1", REFERENCE_URL, REFERENCE_REVISION, cwd=reference)
        command("git", "checkout", "--detach", "FETCH_HEAD", cwd=reference)
    if capture("git", "rev-parse", "HEAD", cwd=reference) != REFERENCE_REVISION:
        raise RuntimeError(f"unexpected reference revision in {reference}")
    return reference


def build_runner(manifest, binary, output, label, lock_directory=None):
    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / "target"))
    lock = manifest.parent / "Cargo.lock"
    if lock_directory:
        shutil.copy2(lock_directory / f"{label}.Cargo.lock", lock)
    cargo_args = ["cargo", "build", "--release", "--manifest-path", str(manifest)]
    if lock_directory:
        cargo_args.append("--locked")
    with (output / f"build-{label}.log").open("w") as log:
        command(*cargo_args, env=env, stdout=log, stderr=subprocess.STDOUT)
    executable = output / label
    shutil.copy2(ROOT / "target/release" / binary, executable)
    shutil.copy2(lock, output / f"{label}.Cargo.lock")
    packages = tomllib.loads(lock.read_text())["package"]
    stack = "; ".join(f'{p["name"]} {p["version"]}' for p in packages
                      if p["name"] in {"cubecl", "wgpu", "wgpu-core", "wgpu-hal", "wgpu-types"})
    return executable, stack


def build_massively(reference, source, output, variant, revision, lock_directory=None):
    upstream = reference / "benchmarks/massively-comparison"
    runners = output / "runners"
    shutil.copytree(upstream / "common", runners / "common", dirs_exist_ok=True)
    runner = runners / variant
    shutil.copytree(upstream / "massively-runner/src", runner / "src")
    cubecl = tomllib.loads((source / "Cargo.toml").read_text())["workspace"]["dependencies"]["cubecl"]["version"]
    manifest = (upstream / "massively-runner/Cargo.toml").read_text()
    # Distinct package names prevent a shared Cargo target directory from
    # reusing another standalone runner's binary and dependency fingerprint.
    binary_name = f"massively-comparison-runner-{variant}"
    manifest = replace_once(manifest, 'name = "massively-comparison-runner"',
                            f'name = {json.dumps(binary_name)}')
    manifest = replace_once(manifest, 'massively = "=0.96.0"',
                            f'massively = {{ path = {json.dumps(str(source / "massively"))} }}')
    manifest = replace_once(manifest, '"=0.11.0-pre.1"', json.dumps("=" + cubecl.lstrip("=")))
    (runner / "Cargo.toml").write_text(manifest + "\n[workspace]\n")
    main = runner / "src/main.rs"
    text = main.read_text()
    for constant, name in [("MASSIVELY_VERSION", "VERSION"), ("MASSIVELY_REVISION", "REVISION")]:
        text = replace_once(text, f"{constant}.into()",
                            f'std::env::var("MASSIVELY_BENCH_IMPLEMENTATION_{name}").unwrap_or_else(|_| {constant}.into())')
    text = replace_once(text, '    println!("{}", serde_json::to_string(&run)?);',
                        '    let mut record = serde_json::to_value(&run)?;\n'
                        f'    record["runner_variant"] = {json.dumps(variant)}.into();\n'
                        '    println!("{}", serde_json::to_string(&record)?);')
    main.write_text(text)
    binary, stack = build_runner(runner / "Cargo.toml", binary_name, output, variant, lock_directory)
    version = tomllib.loads((source / "massively/Cargo.toml").read_text())["package"]["version"]
    return dict(binary=str(binary), version=version, revision=revision,
                source_sha256=source_digest(source), runtime_stack=stack,
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                runner_sha256=hashlib.sha256(main.read_bytes()).hexdigest())


def summarize(records, output):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["config"]["items"], record["config"]["workload"], record["comparison_variant"])].append(record["median_ms"])
    before = any(key[2] == "before" for key in grouped)
    lines = ["# Resident GPU comparison", "",
             f"Reference: [Lampshade `{REFERENCE_REVISION[:12]}`]({REFERENCE_URL}/tree/{REFERENCE_REVISION}).", "",
             "Each value is the median of independent process medians, in milliseconds.",
             "Input upload and validation are excluded; each sample waits for GPU completion.",
             "Reduction includes scalar readback. Massively allocates owned results per call;",
             "Lampshade uses caller-owned output and reusable workspace.", ""]
    lines += ["| Items | Workload | " + ("Before | " if before else "") + "Current | Lampshade | Current / Lampshade |",
              "|---:|---|" + ("---:|" if before else "") + "---:|---:|---:|"]
    for items, workload in sorted({key[:2] for key in grouped}):
        current = statistics.median(grouped[items, workload, "current"])
        reference = statistics.median(grouped[items, workload, "lampshade"])
        prior = f"{statistics.median(grouped[items, workload, 'before']):.3f} | " if before else ""
        lines.append(f"| {items:,} | `{workload}` | {prior}{current:.3f} | {reference:.3f} | {current / reference:.2f}× |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-ref", help="optional local Git revision to measure alongside the working tree")
    parser.add_argument("--lock-directory", type=Path,
                        help="reuse saved <variant>.Cargo.lock files and build with --locked")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000_000, 10_000_000])
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmup-ms", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.sizes + [args.processes, args.samples]) <= 0 or max(args.sizes) > 0xffff_ffff or args.warmup_ms < 0:
        parser.error("sizes, processes, and samples must be positive; sizes must fit u32")
    locks = args.lock_directory.resolve() if args.lock_directory else None
    if locks:
        if not locks.is_relative_to(ROOT):
            parser.error("lock directory must be inside the project")
        names = ["current", "lampshade"] + (["before"] if args.before_ref else [])
        if any(not (locks / f"{name}.Cargo.lock").is_file() for name in names):
            parser.error("lock directory is missing a requested variant's Cargo.lock")
    cache = ROOT / "target/lampshade-comparison"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or cache / stamp).resolve()
    if not output.is_relative_to(ROOT / "target"):
        parser.error("output must be inside the project's target directory")
    output.mkdir(parents=True, exist_ok=False)
    reference = prepare_reference(cache)
    variants = {}
    if args.before_ref:
        revision = capture("git", "rev-parse", f"{args.before_ref}^{{commit}}")
        source = output / "before-source"
        archive = subprocess.check_output(["git", "archive", revision], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        print(f"Building before {revision[:12]}", flush=True)
        variants["before"] = build_massively(reference, source, output, "before", revision, locks)
    revision = capture("git", "rev-parse", "HEAD")
    patch = subprocess.check_output(["git", "diff", "HEAD", "--", "massively", "Cargo.toml"], cwd=ROOT)
    (output / "working-tree.patch").write_bytes(patch)
    untracked = capture("git", "ls-files", "-z", "--others", "--exclude-standard", "massively")
    for name in filter(None, untracked.split("\0")):
        destination = output / "untracked" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    if patch or untracked:
        revision += "-dirty"
    print("Building current Massively", flush=True)
    variants["current"] = build_massively(reference, ROOT, output, "current", revision, locks)
    # The checkout lives below Massively's workspace, so isolate its manifests.
    reference_manifest = reference / "Cargo.toml"
    original = capture("git", "show", "HEAD:Cargo.toml", cwd=reference)
    reference_manifest.write_text(original + "\n[workspace]\n")
    manifest = reference / "benchmarks/massively-comparison/lampshade-runner/Cargo.toml"
    original = capture("git", "show", "HEAD:benchmarks/massively-comparison/lampshade-runner/Cargo.toml", cwd=reference)
    manifest.write_text(original + "\n[workspace]\n")
    print("Building pinned Lampshade", flush=True)
    binary, stack = build_runner(manifest, "lampshade-massively-comparison-runner", output, "lampshade", locks)
    variants["lampshade"] = dict(binary=str(binary), runtime_stack=stack, revision=REFERENCE_REVISION,
                                version=tomllib.loads(reference_manifest.read_text())["package"]["version"],
                                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                                runner_sha256=hashlib.sha256((manifest.parent / "src/main.rs").read_bytes()).hexdigest())
    metadata = dict(created_utc=stamp, rust=capture("rustc", "--version"), variants=variants,
                    lock_directory=str(locks) if locks else None,
                    sizes=args.sizes, processes=args.processes, samples=args.samples, warmup_ms=args.warmup_ms)
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    with (output / "samples.jsonl").open("w") as samples:
        for items in args.sizes:
            for workload in WORKLOADS:
                for process in range(1, args.processes + 1):
                    names = list(variants)
                    rotation = (process - 1) % len(names)
                    for name in names[rotation:] + names[:rotation]:
                        variant = variants[name]
                        env = dict(os.environ, WGPU_BACKEND=os.environ.get("WGPU_BACKEND", "vulkan"),
                                   MASSIVELY_BENCH_ITEMS=str(items), MASSIVELY_BENCH_WORKLOAD=workload,
                                   MASSIVELY_BENCH_WARMUPS="4", MASSIVELY_BENCH_WARMUP_MS=str(args.warmup_ms),
                                   MASSIVELY_BENCH_SAMPLES=str(args.samples), MASSIVELY_BENCH_PROCESS_INDEX=str(process),
                                   MASSIVELY_BENCH_IMPLEMENTATION_VERSION=variant["version"],
                                   MASSIVELY_BENCH_IMPLEMENTATION_REVISION=variant["revision"])
                        with (output / f"{name}-{items}-{workload}-{process}.stderr").open("w") as errors:
                            run = command(variant["binary"], env=env, text=True, stdout=subprocess.PIPE, stderr=errors)
                        record = json.loads(run.stdout)
                        if name != "lampshade" and record.get("runner_variant") != name:
                            raise RuntimeError(f"unexpected runner binary for {name}")
                        if not record["correctness_checked"]:
                            raise RuntimeError("upstream runner did not validate its result")
                        record.update(comparison_variant=name, runtime_stack=variant["runtime_stack"])
                        records.append(record)
                        samples.write(json.dumps(record) + "\n")
                        samples.flush()
                        print(f"{name:9} {items:>11,} {workload:16} process {process}: {record['median_ms']:.3f} ms", flush=True)
    summarize(records, output)
    print(f"Saved comparison to {output}", flush=True)


if __name__ == "__main__":
    main()
