#!/usr/bin/env python3
"""Generate architecture-specific invariance and batched-geometry gate scaffolds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "arch-gate-templates"
ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
ID_RE = re.compile(r"^[a-z0-9]+$")
FAMILIES = {
    "chunk": "chunkinv",
    "tick": "tickinv",
    "batch": "b2geo",
}
RESERVED_SERVER_ENV = {
    "MEMRA_ADDR",
    "MEMRA_MODELS",
    "MEMRA_PP_DEVICES",
    "MEMRA_PP_STAGES",
}


class SpecError(ValueError):
    """The gate specification is incomplete or unsafe to render."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render chunk/tick/batched-geometry gate scripts and fast-gate TSV "
            "fragments from an explicit architecture gate specification."
        )
    )
    parser.add_argument("architecture", help="human-readable architecture name")
    parser.add_argument("artifact", help="default GGUF artifact path")
    parser.add_argument("--spec", required=True, type=Path, help="gate spec JSON")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="output directory (default: tools/generated-arch-gates/<arch-slug>)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace generator-owned files that already exist",
    )
    return parser.parse_args(argv)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise SpecError("architecture must contain at least one letter or digit")
    return slug


def reject_control(value: str, context: str) -> str:
    if not value:
        raise SpecError(f"{context} must not be empty")
    if "\n" in value or "\r" in value or "\t" in value or "\0" in value:
        raise SpecError(f"{context} must not contain control characters")
    return value


def expect_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(f"{context} must be a JSON object")
    return value


def expect_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise SpecError(f"{context} missing required keys: {', '.join(missing)}")
    if unknown:
        raise SpecError(f"{context} has unknown keys: {', '.join(unknown)}")


def expect_string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise SpecError(f"{context} must be a string")
    return reject_control(value, context)


def expect_env(value: Any, context: str) -> str:
    env = expect_string(value, context)
    if not ENV_RE.fullmatch(env):
        raise SpecError(f"{context} must be an environment-variable name")
    return env


def expect_positive_int(value: Any, context: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SpecError(f"{context} must be a {qualifier} integer")
    return value


def expect_int_list(
    value: Any,
    context: str,
    *,
    allow_zero: bool = False,
    minimum_length: int = 1,
) -> list[int]:
    if not isinstance(value, list) or len(value) < minimum_length:
        raise SpecError(f"{context} must be a list with at least {minimum_length} item(s)")
    result = [
        expect_positive_int(item, f"{context}[{index}]", allow_zero=allow_zero)
        for index, item in enumerate(value)
    ]
    if len(set(result)) != len(result):
        raise SpecError(f"{context} must not contain duplicates")
    return result


def expect_string_list(
    value: Any, context: str, *, minimum_length: int = 1
) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum_length:
        raise SpecError(f"{context} must be a list with at least {minimum_length} item(s)")
    result = [
        expect_string(item, f"{context}[{index}]") for index, item in enumerate(value)
    ]
    if len(set(result)) != len(result):
        raise SpecError(f"{context} must not contain duplicates")
    return result


def expect_env_map(value: Any, context: str, *, nonempty: bool) -> dict[str, str]:
    obj = expect_object(value, context)
    if nonempty and not obj:
        raise SpecError(f"{context} must not be empty")
    result: dict[str, str] = {}
    for key, raw_value in obj.items():
        env = expect_env(key, f"{context} key")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (str, int, float)):
            raise SpecError(f"{context}.{env} must be a string or number")
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            raise SpecError(f"{context}.{env} must be finite")
        rendered = str(raw_value)
        reject_control(rendered, f"{context}.{env}")
        result[env] = rendered
    return dict(sorted(result.items()))


def expect_repo_relative(value: Any, context: str) -> str:
    path = expect_string(value, context)
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise SpecError(f"{context} must be a repository-relative path without '..'")
    return path


def expect_regex(value: Any, context: str) -> str:
    pattern = expect_string(value, context)
    try:
        re.compile(pattern)
    except re.error as error:
        raise SpecError(f"{context} is not a valid regex: {error}") from error
    return pattern


def validate_prompt_gate(
    value: Any,
    context: str,
    *,
    values_key: str,
    allow_zero: bool,
    optional_values_key: str | None = None,
) -> dict[str, Any]:
    obj = expect_object(value, context)
    optional = {optional_values_key} if optional_values_key else set()
    expect_keys(
        obj,
        required={"label", "prompts", values_key, "steps", "seam"},
        optional=optional,
        context=context,
    )
    result = {
        "label": expect_string(obj["label"], f"{context}.label"),
        "prompts": [
            expect_repo_relative(item, f"{context}.prompts[{index}]")
            for index, item in enumerate(
                expect_string_list(obj["prompts"], f"{context}.prompts")
            )
        ],
        values_key: expect_int_list(
            obj[values_key], f"{context}.{values_key}", allow_zero=allow_zero
        ),
        "steps": expect_positive_int(obj["steps"], f"{context}.steps"),
        "seam": expect_env(obj["seam"], f"{context}.seam"),
    }
    if optional_values_key:
        raw_optional = obj.get(optional_values_key, [])
        if not isinstance(raw_optional, list):
            raise SpecError(f"{context}.{optional_values_key} must be a list")
        result[optional_values_key] = (
            expect_int_list(
                raw_optional,
                f"{context}.{optional_values_key}",
                allow_zero=False,
            )
            if raw_optional
            else []
        )
    return result


def validate_batch(value: Any) -> dict[str, Any]:
    context = "batch"
    obj = expect_object(value, context)
    expect_keys(
        obj,
        required={
            "model_alias",
            "canary_env",
            "required_gpus",
            "pp_stages",
            "pp_devices",
            "concurrency",
            "port",
            "receipt_dir",
            "server_env",
            "request",
            "liveness",
        },
        optional={"draft_path", "draft_env"},
        context=context,
    )
    alias = expect_string(obj["model_alias"], "batch.model_alias")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", alias):
        raise SpecError("batch.model_alias contains unsupported characters")
    required_gpus = expect_positive_int(obj["required_gpus"], "batch.required_gpus")
    pp_stages = expect_positive_int(obj["pp_stages"], "batch.pp_stages")
    pp_devices = expect_int_list(
        obj["pp_devices"], "batch.pp_devices", allow_zero=True
    )
    if len(pp_devices) != pp_stages:
        raise SpecError("batch.pp_devices length must equal batch.pp_stages")
    if required_gpus < len(set(pp_devices)):
        raise SpecError("batch.required_gpus is smaller than the PP device count")
    if any(device >= required_gpus for device in pp_devices):
        raise SpecError("batch.pp_devices contains an index outside batch.required_gpus")
    concurrency = expect_int_list(obj["concurrency"], "batch.concurrency")
    if any(width < 2 for width in concurrency):
        raise SpecError("batch.concurrency values must all be at least 2")

    server_env = expect_env_map(obj["server_env"], "batch.server_env", nonempty=False)
    reserved = sorted(RESERVED_SERVER_ENV & server_env.keys())
    if reserved:
        raise SpecError(
            "batch.server_env must not override generator-owned variables: "
            + ", ".join(reserved)
        )
    canary_env = expect_env_map(
        obj["canary_env"], "batch.canary_env", nonempty=True
    )
    canary_conflicts = sorted(
        (set(canary_env) & set(server_env)) | (set(canary_env) & RESERVED_SERVER_ENV)
    )
    if canary_conflicts:
        raise SpecError(
            "batch.canary_env conflicts with server or generator-owned variables: "
            + ", ".join(canary_conflicts)
        )

    request = expect_object(obj["request"], "batch.request")
    if not isinstance(request.get("messages"), list) or not request["messages"]:
        raise SpecError("batch.request.messages must be a non-empty list")
    if "model" in request and request["model"] != alias:
        raise SpecError("batch.request.model conflicts with batch.model_alias")
    request = dict(request)
    request["model"] = alias
    try:
        json.dumps(request, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SpecError(f"batch.request is not JSON serializable: {error}") from error

    liveness = expect_object(obj["liveness"], "batch.liveness")
    expect_keys(
        liveness,
        required={"cap_regex", "cap_min", "walk_regex"},
        optional=set(),
        context="batch.liveness",
    )
    cap_regex = expect_regex(liveness["cap_regex"], "batch.liveness.cap_regex")
    if "[0-9]+" not in cap_regex:
        raise SpecError("batch.liveness.cap_regex must include a '[0-9]+' capture")

    draft_path_raw = obj.get("draft_path")
    if draft_path_raw is not None and not isinstance(draft_path_raw, str):
        raise SpecError("batch.draft_path must be a string or null")
    draft_path = (
        reject_control(draft_path_raw, "batch.draft_path") if draft_path_raw else ""
    )
    draft_env_raw = obj.get("draft_env")
    draft_env = (
        expect_env(draft_env_raw, "batch.draft_env")
        if draft_env_raw is not None
        else ""
    )

    port = expect_positive_int(obj["port"], "batch.port")
    if port > 65535:
        raise SpecError("batch.port must be at most 65535")

    return {
        "model_alias": alias,
        "draft_path": draft_path,
        "draft_env": draft_env,
        "canary_env": canary_env,
        "required_gpus": required_gpus,
        "pp_stages": pp_stages,
        "pp_devices": pp_devices,
        "concurrency": concurrency,
        "port": port,
        "receipt_dir": expect_repo_relative(
            obj["receipt_dir"], "batch.receipt_dir"
        ),
        "server_env": server_env,
        "request": request,
        "liveness": {
            "cap_regex": cap_regex,
            "cap_min": expect_positive_int(
                liveness["cap_min"], "batch.liveness.cap_min"
            ),
            "walk_regex": expect_regex(
                liveness["walk_regex"], "batch.liveness.walk_regex"
            ),
        },
    }


def validate_mapping(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SpecError("mapping must be a non-empty list")
    result = []
    for index, raw_row in enumerate(value):
        context = f"mapping[{index}]"
        row = expect_object(raw_row, context)
        expect_keys(
            row,
            required={
                "path_regex",
                "kernel_scope",
                "base_probes",
                "base_spec_probes",
                "gate_families",
            },
            optional=set(),
            context=context,
        )
        families = expect_string_list(
            row["gate_families"], f"{context}.gate_families"
        )
        unknown = sorted(set(families) - FAMILIES.keys())
        if unknown:
            raise SpecError(
                f"{context}.gate_families has unknown values: {', '.join(unknown)}"
            )
        result.append(
            {
                "path_regex": expect_regex(
                    row["path_regex"], f"{context}.path_regex"
                ),
                "kernel_scope": expect_string(
                    row["kernel_scope"], f"{context}.kernel_scope"
                ),
                "base_probes": expect_string_list(
                    row["base_probes"],
                    f"{context}.base_probes",
                    minimum_length=0,
                ),
                "base_spec_probes": expect_string_list(
                    row["base_spec_probes"],
                    f"{context}.base_spec_probes",
                    minimum_length=0,
                ),
                "gate_families": families,
            }
        )
    return result


def validate_spec(raw: Any) -> dict[str, Any]:
    obj = expect_object(raw, "spec")
    expect_keys(
        obj,
        required={"id", "artifact_env", "chunk", "tick", "batch", "mapping"},
        optional=set(),
        context="spec",
    )
    gate_id = expect_string(obj["id"], "spec.id")
    if not ID_RE.fullmatch(gate_id):
        raise SpecError("spec.id must match [a-z0-9]+")
    return {
        "id": gate_id,
        "artifact_env": expect_env(obj["artifact_env"], "spec.artifact_env"),
        "chunk": validate_prompt_gate(
            obj["chunk"],
            "chunk",
            values_key="chunks",
            allow_zero=False,
        ),
        "tick": validate_prompt_gate(
            obj["tick"],
            "tick",
            values_key="budgets",
            allow_zero=True,
            optional_values_key="splits",
        ),
        "batch": validate_batch(obj["batch"]),
        "mapping": validate_mapping(obj["mapping"]),
    }


def render_template(name: str, values: dict[str, str]) -> str:
    template_path = TEMPLATE_DIR / name
    try:
        content = template_path.read_text(encoding="utf-8")
    except OSError as error:
        raise SpecError(f"cannot read template {template_path}: {error}") from error
    placeholders = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", content))
    missing = sorted(placeholders - values.keys())
    extra = sorted(values.keys() - placeholders)
    if missing:
        raise SpecError(f"template {name} missing values for: {', '.join(missing)}")
    if extra:
        raise SpecError(f"template {name} got unused values: {', '.join(extra)}")
    for key, value in values.items():
        content = content.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z0-9_]+\}\}", content):
        raise SpecError(f"template {name} has unreplaced placeholders")
    return content


def shell_literal(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def shell_array_lines(values: dict[str, str], array_name: str) -> str:
    if not values:
        return f"{array_name}=()"
    rendered = " ".join(shlex.quote(f"{key}={value}") for key, value in values.items())
    return f"{array_name}=({rendered})"


def registry_path(out_dir: Path) -> str:
    resolved = out_dir.resolve()
    try:
        path = resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        path = resolved.as_posix()
    if re.search(r"\s", path):
        raise SpecError(
            "output path must not contain whitespace because fast-gate splits commands "
            "on shell words"
        )
    return path


def generated_files(
    architecture: str,
    artifact: str,
    slug: str,
    spec: dict[str, Any],
    out_dir: Path,
) -> dict[str, tuple[str, bool]]:
    artifact_env = spec["artifact_env"]
    chunk = spec["chunk"]
    tick = spec["tick"]
    batch = spec["batch"]
    gate_id = spec["id"]

    chunk_name = f"{slug}-chunk-invariance-gate.sh"
    tick_name = f"{slug}-tick-invariance-gate.sh"
    batch_name = f"{slug}-b2-geometry-gate.sh"
    script_dir = registry_path(out_dir)

    chunk_script = render_template(
        "chunk-wrapper.sh.in",
        {
            "GATE_NAME": f"{slug}-chunk-invariance-gate",
            "ARTIFACT_ENV": artifact_env,
            "DEFAULT_ARTIFACT": shell_literal(artifact),
            "LABEL": shell_literal(chunk["label"]),
            "PROMPTS": shell_literal(",".join(chunk["prompts"])),
            "CHUNKS": shell_literal(",".join(map(str, chunk["chunks"]))),
            "STEPS": str(chunk["steps"]),
            "SEAM": chunk["seam"],
        },
    )
    tick_script = render_template(
        "tick-wrapper.sh.in",
        {
            "GATE_NAME": f"{slug}-tick-invariance-gate",
            "ARTIFACT_ENV": artifact_env,
            "DEFAULT_ARTIFACT": shell_literal(artifact),
            "LABEL": shell_literal(tick["label"]),
            "PROMPTS": shell_literal(",".join(tick["prompts"])),
            "BUDGETS": shell_literal(",".join(map(str, tick["budgets"]))),
            "SPLITS": shell_literal(",".join(map(str, tick["splits"]))),
            "STEPS": str(tick["steps"]),
            "SEAM": tick["seam"],
        },
    )
    batch_script = render_template(
        "b2-geometry.sh.in",
        {
            "GATE_NAME": f"{slug}-b2-geometry-gate",
            "ARCHITECTURE": shell_literal(architecture),
            "ARTIFACT_ENV": artifact_env,
            "DEFAULT_ARTIFACT": shell_literal(artifact),
            "DRAFT_ENV": shell_literal(batch["draft_env"]),
            "DEFAULT_DRAFT": shell_literal(batch["draft_path"]),
            "MODEL_ALIAS": shell_literal(batch["model_alias"]),
            "REQUIRED_GPUS": str(batch["required_gpus"]),
            "PP_STAGES": str(batch["pp_stages"]),
            "PP_DEVICES": shell_literal(",".join(map(str, batch["pp_devices"]))),
            "CONCURRENCY": " ".join(map(str, batch["concurrency"])),
            "DEFAULT_PORT": str(batch["port"]),
            "RECEIPT_DIR": shell_literal(batch["receipt_dir"]),
            "SERVER_ENV": shell_array_lines(batch["server_env"], "SERVER_ENV"),
            "CANARY_ENV": shell_array_lines(batch["canary_env"], "CANARY_ENV"),
            "REQUEST_BODY": shell_literal(
                json.dumps(
                    batch["request"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
            "CAP_REGEX": shell_literal(batch["liveness"]["cap_regex"]),
            "CAP_MIN": str(batch["liveness"]["cap_min"]),
            "WALK_REGEX": shell_literal(batch["liveness"]["walk_regex"]),
        },
    )

    model_rows = [
        "# Generated fragment: review, then merge these rows into tools/fast-gate/models.tsv.",
        "# The generator deliberately does not edit the canonical registry.",
    ]
    for family, prefix in FAMILIES.items():
        script_name = {
            "chunk": chunk_name,
            "tick": tick_name,
            "batch": batch_name,
        }[family]
        command = f"{script_dir}/{script_name}"
        for canary in (False, True):
            model_rows.append(
                render_template(
                    "models-row.tsv.in",
                    {
                        "ID": f"{prefix}{gate_id}{'c' if canary else ''}",
                        "SCRIPT": command,
                        "ARGS": "--canary" if canary else "-",
                    },
                ).rstrip("\n")
            )

    map_rows = [
        "# Generated fragment: replace or merge matching rows in tools/fast-gate/map.tsv.",
        "# Each row retains the declared base probes and appends the new architecture gates.",
    ]
    for row in spec["mapping"]:
        probes = list(row["base_probes"])
        for family in row["gate_families"]:
            prefix = FAMILIES[family]
            probes.extend((f"{prefix}{gate_id}", f"{prefix}{gate_id}c"))
        probes = list(dict.fromkeys(probes))
        map_rows.append(
            render_template(
                "map-row.tsv.in",
                {
                    "PATH_REGEX": row["path_regex"],
                    "KERNEL_SCOPE": row["kernel_scope"],
                    "PROBES": ",".join(probes) if probes else "-",
                    "SPEC_PROBES": (
                        ",".join(row["base_spec_probes"])
                        if row["base_spec_probes"]
                        else "-"
                    ),
                },
            ).rstrip("\n")
        )

    normalized = {
        "schema_version": 1,
        "architecture": architecture,
        "artifact": artifact,
        "spec": spec,
    }
    spec_text = json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    checksum_rows = {
        chunk_name: hashlib.sha256(chunk_script.encode()).hexdigest(),
        tick_name: hashlib.sha256(tick_script.encode()).hexdigest(),
        batch_name: hashlib.sha256(batch_script.encode()).hexdigest(),
        "fast-gate-models.tsv": hashlib.sha256(
            ("\n".join(model_rows) + "\n").encode()
        ).hexdigest(),
        "fast-gate-map.tsv": hashlib.sha256(
            ("\n".join(map_rows) + "\n").encode()
        ).hexdigest(),
    }
    normalized["generated_sha256"] = checksum_rows
    spec_text = json.dumps(normalized, indent=2, sort_keys=True) + "\n"

    return {
        chunk_name: (chunk_script, True),
        tick_name: (tick_script, True),
        batch_name: (batch_script, True),
        "fast-gate-models.tsv": ("\n".join(model_rows) + "\n", False),
        "fast-gate-map.tsv": ("\n".join(map_rows) + "\n", False),
        "gate-spec.json": (spec_text, False),
    }


def write_files(
    out_dir: Path, files: dict[str, tuple[str, bool]], *, force: bool
) -> None:
    existing = sorted(name for name in files if (out_dir / name).exists())
    if existing and not force:
        raise SpecError(
            "refusing to overwrite generator-owned files without --force: "
            + ", ".join(existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (content, executable) in files.items():
        destination = out_dir / name
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        mode = temporary.stat().st_mode
        if executable:
            temporary.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        architecture = reject_control(args.architecture, "architecture")
        artifact = reject_control(args.artifact, "artifact")
        slug = slugify(architecture)
        raw = json.loads(args.spec.read_text(encoding="utf-8"))
        spec = validate_spec(raw)
        out_dir = args.out_dir or ROOT / "tools" / "generated-arch-gates" / slug
        files = generated_files(architecture, artifact, slug, spec, out_dir)
        write_files(out_dir, files, force=args.force)
    except (OSError, json.JSONDecodeError, SpecError) as error:
        print(f"generate-arch-gates: ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"generate-arch-gates: wrote {len(files)} files for {architecture} to {out_dir}"
    )
    print(
        "generate-arch-gates: review fast-gate-models.tsv and fast-gate-map.tsv; "
        "the canonical registries were not modified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
