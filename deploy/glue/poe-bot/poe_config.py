#!/usr/bin/env python3
"""Dependency-free configuration validation for the Poe shim."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse


UNSET = {
    "",
    "change-me",
    "replace-me",
    "replace-with-32-character-access-key",
    "replace-with-memra-key",
}
PATH_RE = re.compile(r"^/[A-Za-z0-9/_-]*$")


class ConfigError(ValueError):
    pass


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be a number") from error
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _secret(name: str, *, placeholder: str | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if not value and placeholder is not None:
        value = placeholder
    if value.lower() in UNSET:
        raise ConfigError(f"{name} still contains an unset/default value")
    if "\n" in value or "\r" in value:
        raise ConfigError(f"{name} must be a single line")
    return value


def _validate_backend_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        loopback_http = (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port is not None
        )
        public_https = parsed.scheme == "https" and bool(parsed.hostname)
        invalid = (
            not (loopback_http or public_https)
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
            "MEMRA_POE_BACKEND_URL must be a loopback HTTP or public HTTPS origin"
        )
    return value.rstrip("/")


@dataclass(frozen=True)
class PoeConfig:
    poe_access_key: str = field(repr=False)
    backend_key: str = field(repr=False)
    backend_url: str = "http://127.0.0.1:8002"
    model: str = "stepfun/step-3.7-flash"
    path: str = "/poe"
    max_input_chars: int = 60000
    max_messages: int = 64
    max_output_tokens: int = 512
    max_concurrency: int = 1
    queue_wait_seconds: float = 5.0
    backend_timeout_seconds: float = 300.0

    @classmethod
    def from_env(cls, *, allow_placeholders: bool = False) -> "PoeConfig":
        poe_placeholder = "x" * 32 if allow_placeholders else None
        backend_placeholder = "mk-dry-run-backend-key" if allow_placeholders else None
        poe_access_key = _secret("POE_ACCESS_KEY", placeholder=poe_placeholder)
        backend_key = _secret(
            "MEMRA_POE_BACKEND_KEY", placeholder=backend_placeholder
        )
        if len(poe_access_key) != 32:
            raise ConfigError("POE_ACCESS_KEY must be exactly 32 characters")

        model = os.environ.get(
            "MEMRA_POE_MODEL", "stepfun/step-3.7-flash"
        ).strip()
        if not model or len(model) > 200:
            raise ConfigError("MEMRA_POE_MODEL must be 1-200 characters")

        path = os.environ.get("MEMRA_POE_PATH", "/poe").strip()
        if (
            not PATH_RE.fullmatch(path)
            or path == "/"
            or "//" in path
            or path.endswith("/")
        ):
            raise ConfigError(
                "MEMRA_POE_PATH must be a non-root path without a trailing slash"
            )

        return cls(
            poe_access_key=poe_access_key,
            backend_key=backend_key,
            backend_url=_validate_backend_url(
                os.environ.get(
                    "MEMRA_POE_BACKEND_URL", "http://127.0.0.1:8002"
                ).strip()
            ),
            model=model,
            path=path,
            max_input_chars=_env_int(
                "MEMRA_POE_MAX_INPUT_CHARS", 60000, 1000, 200000
            ),
            max_messages=_env_int("MEMRA_POE_MAX_MESSAGES", 64, 1, 128),
            max_output_tokens=_env_int(
                "MEMRA_POE_MAX_OUTPUT_TOKENS", 512, 1, 2048
            ),
            max_concurrency=_env_int("MEMRA_POE_MAX_CONCURRENCY", 1, 1, 2),
            queue_wait_seconds=_env_float(
                "MEMRA_POE_QUEUE_WAIT_SECONDS", 5.0, 0.1, 30.0
            ),
            backend_timeout_seconds=_env_float(
                "MEMRA_POE_BACKEND_TIMEOUT_SECONDS", 300.0, 30.0, 600.0
            ),
        )


def print_config(config: PoeConfig, mode: str) -> None:
    print(f"PASS: Poe bot config {mode}")
    print(f"  protocol path: {config.path}")
    print(f"  backend:       {config.backend_url}")
    print(f"  model:         {config.model}")
    print(
        "  caps:          "
        f"concurrency={config.max_concurrency} "
        f"messages={config.max_messages} "
        f"input_chars={config.max_input_chars} "
        f"max_output={config.max_output_tokens}"
    )
    print("  pricing:       no rate card (free bot)")
    print("  secrets:       present and redacted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate with placeholder secrets and make no connections",
    )
    args = parser.parse_args()
    try:
        config = PoeConfig.from_env(allow_placeholders=args.dry_run)
    except ConfigError as error:
        parser.exit(1, f"error: {error}\n")
    print_config(config, "dry run is valid" if args.dry_run else "is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
