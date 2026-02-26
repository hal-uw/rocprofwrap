#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
from typing import List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run amd_smi_query alongside an app. "
            "Pass --device as a comma-separated list; one sampler thread "
            "is launched per GPU inside the binary."
        )
    )
    parser.add_argument(
        "-d", "--devices",
        required=True,
        help='GPU device id(s), e.g. "0" or "0,1,2,3"',
    )
    parser.add_argument(
        "-p", "--prefix",
        required=True,
        help=(
            "Prefix for output file names. "
            "Files will be named <prefix>-gpu<id>.bin"
        ),
    )
    parser.add_argument(
        "--query",
        default=None,
        help=(
            "Path to amd_smi_query binary "
            "(default: ./amd_smi_query next to this script)."
        ),
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=1,
        help="Sampling interval in ms (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files (default: current directory).",
    )
    parser.add_argument(
        "--ring-capacity",
        type=int,
        default=262144,
        help="Ring capacity per GPU in records (default: 262144).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Request realtime scheduling in the sampler (best effort).",
    )
    parser.add_argument(
        "--mlock",
        action="store_true",
        help="Request mlockall() in the sampler (best effort).",
    )
    parser.add_argument(
        "--post-convert-csv",
        action="store_true",
        help="After the run, convert each .bin file to .csv.",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help='Command to run after "--", e.g. -- python3 app.py',
    )
    return parser


def ensure_executable(path: str, desc: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} not found: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{desc} is not executable: {path}")


def terminate_process(proc: subprocess.Popen, timeout_sec: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.cmd or args.cmd[0] != "--":
        parser.error('A command is required and must follow "--".')
    app_cmd = args.cmd[1:]
    if not app_cmd:
        parser.error("No command provided after --.")

    if args.interval_ms <= 0:
        parser.error("--interval-ms must be > 0")
    if args.ring_capacity <= 0:
        parser.error("--ring-capacity must be > 0")

    # Validate device list (integers only, no parsing beyond that —
    # the C++ binary owns the actual validation).
    for part in args.devices.split(","):
        part = part.strip()
        if part and not part.isdigit():
            parser.error(f"Invalid device id: {part!r}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    query_path = args.query or os.path.join(script_dir, "amd_smi_query")
    try:
        ensure_executable(query_path, "amd_smi_query")
    except (FileNotFoundError, PermissionError) as exc:
        parser.error(str(exc))

    os.makedirs(args.output_dir, exist_ok=True)

    # Output prefix includes the output directory so the C++ binary writes
    # files directly into the right place.
    output_prefix = os.path.abspath(
        os.path.join(args.output_dir, args.prefix)
    )

    sampler_cmd = [
        query_path,
        "--device",          args.devices,
        "--output-prefix",   output_prefix,
        "--interval-us",     str(args.interval_ms * 1000),
        "--ring-capacity",   str(args.ring_capacity),
    ]
    if args.realtime:
        sampler_cmd.append("--realtime")
    if args.mlock:
        sampler_cmd.append("--mlock")

    # Derive the list of .bin paths for optional post-conversion.
    device_ids = [p.strip() for p in args.devices.split(",") if p.strip()]
    bin_paths = [
        f"{output_prefix}-gpu{dev}.bin"
        for dev in device_ids
    ]

    sampler_proc = subprocess.Popen(sampler_cmd)

    def shutdown_sampler() -> None:
        terminate_process(sampler_proc, timeout_sec=5.0)

    try:
        app_proc = subprocess.Popen(app_cmd)

        try:
            app_rc = app_proc.wait()
        except KeyboardInterrupt:
            if app_proc.poll() is None:
                app_proc.send_signal(signal.SIGINT)
                try:
                    app_proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    app_proc.kill()
                    app_proc.wait()
            app_rc = app_proc.returncode if app_proc.returncode is not None else 130

        shutdown_sampler()

        sampler_rc = sampler_proc.returncode
        if sampler_rc not in (0, 130, -signal.SIGINT):
            print(
                f"[WARN] amd_smi_query exited with code {sampler_rc}",
                file=sys.stderr,
            )

        if args.post_convert_csv:
            for bin_path in bin_paths:
                if not os.path.exists(bin_path):
                    print(
                        f"[WARN] expected output not found: {bin_path}", file=sys.stderr)
                    continue
                csv_path = os.path.splitext(bin_path)[0] + ".csv"
                rc = subprocess.call(
                    [query_path, "--convert", bin_path, "--csv", csv_path])
                if rc != 0:
                    print(
                        f"[WARN] conversion failed for {bin_path}, rc={rc}", file=sys.stderr)

        return app_rc if app_rc is not None else 0

    finally:
        shutdown_sampler()


if __name__ == "__main__":
    sys.exit(main())
