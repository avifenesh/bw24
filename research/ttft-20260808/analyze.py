#!/usr/bin/env python3
"""Join client TTFT rows to memra-server phase traces by request id."""

import argparse
import json
import pathlib
import shlex

PHASE_FIELDS = (
    "request_parse_ms",
    "admission_ms",
    "queue_wait_ms",
    "tokenize_ms",
    "prime_wait_ms",
    "prime_ms",
    "decode_wait_ms",
    "sse_handoff_ms",
    "total_ms",
)


def upper_median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def parse_trace(line):
    if not line.startswith("[ttft] "):
        return None
    fields = {}
    for token in shlex.split(line[len("[ttft] ") :]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    if fields.get("outcome") != "first_sse_byte":
        return None
    for field in PHASE_FIELDS:
        raw = fields.get(field)
        fields[field] = None if raw in (None, "na") else float(raw)
    fields["prompt_tokens"] = int(fields["prompt_tokens"])
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", required=True)
    parser.add_argument(
        "--client",
        action="append",
        required=True,
        metavar="SHAPE=JSONL",
    )
    parser.add_argument("--joined", required=True)
    parser.add_argument("--table", required=True)
    args = parser.parse_args()

    traces = {}
    for line in pathlib.Path(args.server_log).read_text(errors="replace").splitlines():
        trace = parse_trace(line)
        if trace:
            traces[trace["id"]] = trace

    joined = []
    for spec in args.client:
        shape, path = spec.split("=", 1)
        for line in pathlib.Path(path).read_text().splitlines():
            client = json.loads(line)
            if not client["measured"]:
                continue
            trace = traces.get(client["id"])
            if trace is None:
                raise RuntimeError(f"missing [ttft] trace for request {client['id']}")
            row = {**client, **trace, "shape": shape}
            row["client_minus_server_ms"] = (
                row["client_ttft_ms"] - row["total_ms"]
            )
            joined.append(row)

    joined_path = pathlib.Path(args.joined)
    joined_path.parent.mkdir(parents=True, exist_ok=True)
    with joined_path.open("w") as output:
        for row in joined:
            output.write(json.dumps(row, sort_keys=True) + "\n")

    columns = (
        "shape",
        "n",
        "prompt_tokens",
        "client_ttft_p50_ms",
        *PHASE_FIELDS,
        "client_minus_server_ms",
    )
    summaries = []
    for shape in sorted({row["shape"] for row in joined}):
        rows = [row for row in joined if row["shape"] == shape]
        summary = {
            "shape": shape,
            "n": len(rows),
            "prompt_tokens": sorted({row["prompt_tokens"] for row in rows}),
            "client_ttft_p50_ms": upper_median(
                [row["client_ttft_ms"] for row in rows]
            ),
        }
        if len(summary["prompt_tokens"]) != 1:
            raise RuntimeError(
                f"{shape}: inconsistent prompt tokens {summary['prompt_tokens']}"
            )
        summary["prompt_tokens"] = summary["prompt_tokens"][0]
        for field in PHASE_FIELDS:
            values = [row[field] for row in rows if row[field] is not None]
            if len(values) != len(rows):
                raise RuntimeError(f"{shape}: missing {field} in one or more traces")
            summary[field] = upper_median(values)
        summary["client_minus_server_ms"] = upper_median(
            [row["client_minus_server_ms"] for row in rows]
        )
        summaries.append(summary)

    table_path = pathlib.Path(args.table)
    with table_path.open("w") as output:
        output.write("\t".join(columns) + "\n")
        for summary in summaries:
            output.write(
                "\t".join(
                    str(summary[column])
                    if column in ("shape", "n", "prompt_tokens")
                    else f"{summary[column]:.3f}"
                    for column in columns
                )
                + "\n"
            )
    print(table_path.read_text(), end="")


if __name__ == "__main__":
    main()
