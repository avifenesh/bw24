#!/usr/bin/env python3
"""Poe server-bot adapter for memra's OpenAI-compatible chat endpoint."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator

import fastapi_poe as fp
import httpx
from fastapi import FastAPI
from poe_config import PoeConfig


LOGGER = logging.getLogger("memra.poe")
ROLE_MAP = {
    "system": "system",
    "user": "user",
    "bot": "assistant",
}


class PoeInputError(ValueError):
    def __init__(self, message: str, error_type: str = "user_caused_error") -> None:
        super().__init__(message)
        self.error_type = error_type


class BackendError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def _conversation_key(request: fp.QueryRequest) -> str:
    material = f"poe\0{request.user_id}\0{request.conversation_id}".encode("utf-8")
    return f"poe-{hashlib.sha256(material).hexdigest()[:32]}"


def _build_messages(request: fp.QueryRequest, config: PoeConfig) -> list[dict[str, str]]:
    if request.tools or request.tool_calls or request.tool_results:
        raise PoeInputError("Tool calls are not enabled for this research preview.")
    if len(request.query) > config.max_messages:
        raise PoeInputError(
            "This conversation has too many turns for the preview.",
            "user_message_too_long",
        )

    messages: list[dict[str, str]] = []
    input_chars = 0
    saw_user = False
    for message in request.query:
        if message.attachments:
            raise PoeInputError("Attachments are not enabled for this bot.")
        if message.role == "system" and request.skip_system_prompt:
            continue
        role = ROLE_MAP.get(message.role)
        if role is None:
            raise PoeInputError(f"Unsupported Poe message role: {message.role}")
        content = message.content
        input_chars += len(content)
        if input_chars > config.max_input_chars:
            raise PoeInputError(
                "This conversation is too long for the preview.",
                "user_message_too_long",
            )
        messages.append({"role": role, "content": content})
        saw_user = saw_user or role == "user"

    if not messages or not saw_user:
        raise PoeInputError("A user message is required.")
    return messages


def _build_payload(request: fp.QueryRequest, config: PoeConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": config.model,
        "messages": _build_messages(request, config),
        "max_tokens": config.max_output_tokens,
        "stream": True,
        "include_reasoning": False,
    }
    conversation_key = _conversation_key(request)
    payload["cache_salt"] = conversation_key
    payload["session_id"] = conversation_key

    if request.temperature is not None:
        if not 0 <= request.temperature <= 2:
            raise PoeInputError("Temperature must be between 0 and 2.")
        payload["temperature"] = request.temperature
    if request.stop_sequences:
        if len(request.stop_sequences) > 8 or any(
            len(stop) > 256 for stop in request.stop_sequences
        ):
            raise PoeInputError("Too many or overly long stop sequences.")
        payload["stop"] = request.stop_sequences
    return payload


async def _iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


class MemraPoeBot(fp.PoeBot):
    def __init__(
        self,
        config: PoeConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(path=config.path)
        self.config = config
        self.transport = transport
        self.semaphore = asyncio.Semaphore(config.max_concurrency)

    async def get_settings(
        self, setting: fp.SettingsRequest
    ) -> fp.SettingsResponse:
        del setting
        return fp.SettingsResponse(
            response_version=2,
            allow_attachments=False,
            expand_text_attachments=False,
            enable_image_comprehension=False,
            enforce_author_role_alternation=False,
            enable_multi_entity_prompting=False,
            introduction_message=(
                "Step-3.7-Flash research preview served by memra. "
                "Availability is limited to the trial window."
            ),
            rate_card=None,
            cost_label=None,
            parameter_controls=None,
        )

    async def _stream_backend(
        self, request: fp.QueryRequest
    ) -> AsyncIterator[str]:
        payload = _build_payload(request, self.config)
        timeout = httpx.Timeout(
            connect=10.0,
            read=self.config.backend_timeout_seconds,
            write=15.0,
            pool=5.0,
        )
        headers = {
            "Authorization": f"Bearer {self.config.backend_key}",
            "Content-Type": "application/json",
            "User-Agent": "memra-poe-bot/1",
        }
        endpoint = f"{self.config.backend_url}/v1/chat/completions"

        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self.transport,
        ) as client:
            async with client.stream(
                "POST", endpoint, headers=headers, json=payload
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    LOGGER.warning(
                        "memra backend rejected Poe request status=%s",
                        response.status_code,
                    )
                    if response.status_code == 429:
                        raise BackendError(
                            "The research preview is busy. Retry shortly.",
                            retryable=True,
                        )
                    if response.status_code in {401, 403}:
                        raise BackendError(
                            "The preview backend is not configured correctly.",
                            retryable=False,
                        )
                    if 400 <= response.status_code < 500:
                        raise BackendError(
                            "The preview could not accept this request.",
                            retryable=False,
                        )
                    raise BackendError(
                        "The preview backend is temporarily unavailable.",
                        retryable=True,
                    )

                saw_done = False
                async for data in _iter_sse_data(response):
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as error:
                        raise BackendError(
                            "The preview returned an invalid stream.",
                            retryable=True,
                        ) from error
                    if "error" in event:
                        raise BackendError(
                            "The preview interrupted this response.",
                            retryable=True,
                        )
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        raise BackendError(
                            "The preview returned an invalid response.",
                            retryable=True,
                        )
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield content
                if not saw_done:
                    raise BackendError(
                        "The preview stream ended early.",
                        retryable=True,
                    )

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterator[fp.PartialResponse]:
        acquired = False
        sent_text = False
        try:
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.config.queue_wait_seconds,
            )
            acquired = True
            async for text in self._stream_backend(request):
                sent_text = True
                yield fp.PartialResponse(text=text)
            if not sent_text:
                yield fp.ErrorResponse(
                    text="The preview returned no visible text. Retry once.",
                    allow_retry=True,
                )
        except PoeInputError as error:
            yield fp.ErrorResponse(
                text=str(error),
                allow_retry=False,
                error_type=error.error_type,
            )
        except asyncio.TimeoutError:
            yield fp.ErrorResponse(
                text="The research preview is busy. Retry shortly.",
                allow_retry=True,
            )
        except httpx.TimeoutException:
            LOGGER.warning("memra backend timed out")
            yield fp.ErrorResponse(
                text="The preview timed out. Retry shortly.",
                allow_retry=True,
            )
        except httpx.HTTPError as error:
            LOGGER.warning("memra backend transport error type=%s", type(error).__name__)
            yield fp.ErrorResponse(
                text="The preview backend is temporarily unavailable.",
                allow_retry=True,
            )
        except BackendError as error:
            yield fp.ErrorResponse(
                text=str(error),
                allow_retry=error.retryable,
            )
        finally:
            if acquired:
                self.semaphore.release()


def create_app(
    config: PoeConfig | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    config = config or PoeConfig.from_env()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    bot = MemraPoeBot(config, transport=transport)

    @app.get(f"{config.path}/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    return fp.make_app(bot, access_key=config.poe_access_key, app=app)
