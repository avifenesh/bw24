#!/usr/bin/env python3
"""Render and validate the trial's AI Horde bridge configuration offline."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from urllib.parse import urlparse


ADVERTISED_MODEL = "memra-research-preview"
DEFAULT_OUTPUT = Path("/var/lib/memra-horde-worker/config.yaml")
TEMPLATE = Path(__file__).with_name("config.template.yaml")
PLACEHOLDER_RE = re.compile(r"@@[A-Z0-9_]+@@")
WORKER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,99}$")
UNSET_SECRETS = {
    "",
    "0000000000",
    "change-me",
    "replace-me",
    "replace-with-ai-horde-key",
    "replace-with-memra-key",
}


class ConfigError(ValueError):
    pass


def env_text(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ConfigError(f"{name} is required")
    value = value.strip()
    if not value:
        raise ConfigError(f"{name} must not be empty")
    if "\n" in value or "\r" in value:
        raise ConfigError(f"{name} must be a single line")
    return value


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def validate_cluster_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        invalid = (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        )
    except ValueError:
        invalid = True
    if invalid:
        raise ConfigError(
            "MEMRA_HORDE_CLUSTER_URL must be an HTTPS origin without credentials or a path"
        )
    return value.rstrip("/")


def validate_server_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        invalid = (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        )
    except ValueError:
        invalid = True
    if invalid:
        raise ConfigError(
            "MEMRA_HORDE_SERVER_URL must be a loopback HTTP origin with an explicit port"
        )
    return value.rstrip("/")


def validate_secret(name: str, value: str) -> str:
    if value.lower() in UNSET_SECRETS:
        raise ConfigError(f"{name} still contains an unset/default value")
    return value


def require_integer(
    values: dict[str, object], key: str, minimum: int, maximum: int
) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise ConfigError(f"{key} must stay between {minimum} and {maximum}")
    return value


def parse_top_level(text: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.lstrip().startswith("#") or line[0].isspace():
            continue
        if ":" not in line:
            raise ConfigError(f"invalid top-level YAML at line {line_number}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        if key in parsed:
            raise ConfigError(f"duplicate top-level key: {key}")
        try:
            parsed[key] = json.loads(raw_value)
        except json.JSONDecodeError as error:
            raise ConfigError(
                f"template value for {key} is not JSON-compatible YAML"
            ) from error
    return parsed


def validate_rendered(text: str) -> dict[str, object]:
    unresolved = sorted(set(PLACEHOLDER_RE.findall(text)))
    if unresolved:
        raise ConfigError(f"unresolved template placeholders: {', '.join(unresolved)}")

    values = parse_top_level(text)
    required = {
        "AiHordeApiKey",
        "workerName",
        "clusterUrl",
        "serverEngine",
        "serverUrl",
        "serverApiKey",
        "openaiCompatMode",
        "serverModel",
        "model",
        "ctx",
        "maxLength",
        "enforceCtxLimit",
        "threads",
        "timeout",
        "refreshTime",
        "nsfw",
        "enableCsamFilter",
        "outputPrompt",
        "logFile",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ConfigError(f"rendered config is missing: {', '.join(missing)}")

    validate_secret("AiHordeApiKey", str(values["AiHordeApiKey"]))
    validate_secret("serverApiKey", str(values["serverApiKey"]))
    worker_name = str(values["workerName"])
    if not WORKER_NAME_RE.fullmatch(worker_name):
        raise ConfigError("rendered workerName is invalid")
    if not str(values["serverModel"]).strip():
        raise ConfigError("rendered serverModel must not be empty")
    if values["serverEngine"] != "openaicompat":
        raise ConfigError("serverEngine must remain openaicompat")
    if values["openaiCompatMode"] != "text":
        raise ConfigError("openaiCompatMode must remain text")
    if values["model"] != ADVERTISED_MODEL:
        raise ConfigError(f"model must remain {ADVERTISED_MODEL}")
    if values["enforceCtxLimit"] != "enabled":
        raise ConfigError("enforceCtxLimit must remain enabled")
    if values["outputPrompt"] != "" or values["logFile"] != "":
        raise ConfigError("prompt and bridge file logging must remain disabled")

    validate_cluster_url(str(values["clusterUrl"]))
    validate_server_url(str(values["serverUrl"]))

    require_integer(values, "ctx", 1024, 32768)
    require_integer(values, "maxLength", 1, 1024)
    require_integer(values, "threads", 1, 2)
    require_integer(values, "timeout", 30, 600)
    require_integer(values, "refreshTime", 1000, 60000)
    if values["nsfw"] not in {"enabled", "disabled"}:
        raise ConfigError("nsfw must be enabled or disabled")
    return values


def render_config(allow_placeholder_secrets: bool) -> tuple[str, dict[str, object]]:
    if not TEMPLATE.is_file():
        raise ConfigError(f"config template not found: {TEMPLATE}")

    horde_key = os.environ.get("AI_HORDE_API_KEY", "").strip()
    backend_key = os.environ.get("MEMRA_HORDE_BACKEND_KEY", "").strip()
    if allow_placeholder_secrets:
        horde_key = horde_key or "dry-run-horde-key"
        backend_key = backend_key or "mk-dry-run-backend-key"
    else:
        horde_key = validate_secret("AI_HORDE_API_KEY", horde_key)
        backend_key = validate_secret("MEMRA_HORDE_BACKEND_KEY", backend_key)

    worker_name = env_text(
        "MEMRA_HORDE_WORKER_NAME", "memra-research-preview-runpod"
    )
    if not WORKER_NAME_RE.fullmatch(worker_name):
        raise ConfigError(
            "MEMRA_HORDE_WORKER_NAME must be 3-100 characters using letters, "
            "digits, dot, underscore, colon, or hyphen"
        )

    server_model = env_text(
        "MEMRA_HORDE_SERVER_MODEL", "stepfun/step-3.7-flash"
    )
    if len(server_model) > 200:
        raise ConfigError("MEMRA_HORDE_SERVER_MODEL must be at most 200 characters")

    nsfw_mode = env_text("MEMRA_HORDE_NSFW", "disabled").lower()
    if nsfw_mode not in {"enabled", "disabled"}:
        raise ConfigError("MEMRA_HORDE_NSFW must be enabled or disabled")

    substitutions: dict[str, str] = {
        "@@AI_HORDE_API_KEY@@": json.dumps(horde_key),
        "@@WORKER_NAME@@": json.dumps(worker_name),
        "@@CLUSTER_URL@@": json.dumps(
            validate_cluster_url(
                env_text("MEMRA_HORDE_CLUSTER_URL", "https://aihorde.net")
            )
        ),
        "@@SERVER_URL@@": json.dumps(
            validate_server_url(
                env_text("MEMRA_HORDE_SERVER_URL", "http://127.0.0.1:8002")
            )
        ),
        "@@BACKEND_KEY@@": json.dumps(backend_key),
        "@@SERVER_MODEL@@": json.dumps(server_model),
        "@@CONTEXT_TOKENS@@": str(
            env_int("MEMRA_HORDE_CONTEXT_TOKENS", 16384, 1024, 32768)
        ),
        "@@MAX_OUTPUT_TOKENS@@": str(
            env_int("MEMRA_HORDE_MAX_OUTPUT_TOKENS", 512, 1, 1024)
        ),
        "@@THREADS@@": str(env_int("MEMRA_HORDE_THREADS", 1, 1, 2)),
        "@@TIMEOUT_SECONDS@@": str(
            env_int("MEMRA_HORDE_TIMEOUT_SECONDS", 300, 30, 600)
        ),
        "@@REFRESH_MILLISECONDS@@": str(
            env_int("MEMRA_HORDE_REFRESH_MILLISECONDS", 5000, 1000, 60000)
        ),
        "@@NSFW_MODE@@": json.dumps(nsfw_mode),
    }

    rendered = TEMPLATE.read_text(encoding="utf-8")
    for placeholder, value in substitutions.items():
        rendered = rendered.replace(placeholder, value)
    values = validate_rendered(rendered)
    return rendered, values


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        path.chmod(0o600)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def check_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ConfigError(f"rendered config not found: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ConfigError(f"rendered config must not be group/world-readable: {mode:o}")
    return validate_rendered(path.read_text(encoding="utf-8"))


def print_summary(values: dict[str, object], mode: str) -> None:
    print(f"PASS: Horde bridge config {mode}")
    print(f"  advertised model: {values['model']}")
    print(f"  backend model:    {values['serverModel']}")
    print(f"  worker:           {values['workerName']}")
    print(f"  backend:          {values['serverUrl']}")
    print(
        "  caps:             "
        f"threads={values['threads']} context={values['ctx']} "
        f"max_output={values['maxLength']}"
    )
    print("  secrets:          present and redacted")


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--dry-run",
        action="store_true",
        help="validate with placeholder secrets; do not write",
    )
    actions.add_argument(
        "--validate",
        action="store_true",
        help="validate the current environment with real secrets; do not write",
    )
    actions.add_argument(
        "--check",
        type=Path,
        metavar="PATH",
        help="validate an existing rendered config without connecting",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        if args.check:
            values = check_file(args.check)
            print_summary(values, f"file is valid: {args.check}")
            return 0

        rendered, values = render_config(allow_placeholder_secrets=args.dry_run)
        if args.dry_run:
            print_summary(values, "dry run is valid")
            return 0
        if args.validate:
            print_summary(values, "environment is valid")
            return 0

        write_atomic(args.output, rendered)
        print_summary(values, f"written to {args.output}")
        return 0
    except (ConfigError, OSError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
