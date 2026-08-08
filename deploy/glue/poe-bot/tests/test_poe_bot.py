from __future__ import annotations

import json

import fastapi_poe as fp
import httpx
import pytest

from poe_bot import MemraPoeBot, PoeConfig, create_app


def make_config(**overrides: object) -> PoeConfig:
    values: dict[str, object] = {
        "poe_access_key": "p" * 32,
        "backend_key": "mk-poe-test-key",
        "backend_url": "http://memra.test",
        "model": "stepfun/step-3.7-flash",
        "path": "/poe",
        "max_input_chars": 60000,
        "max_messages": 64,
        "max_output_tokens": 512,
        "max_concurrency": 1,
        "queue_wait_seconds": 1.0,
        "backend_timeout_seconds": 10.0,
    }
    values.update(overrides)
    return PoeConfig(**values)


def make_request(
    query: list[fp.ProtocolMessage] | None = None,
    **overrides: object,
) -> fp.QueryRequest:
    values: dict[str, object] = {
        "version": "1.0",
        "type": "query",
        "query": query
        or [
            fp.ProtocolMessage(role="system", content="Be concise."),
            fp.ProtocolMessage(role="user", content="Hello"),
            fp.ProtocolMessage(role="bot", content="Hi"),
            fp.ProtocolMessage(role="user", content="Continue"),
        ],
        "user_id": "poe-user-secret",
        "conversation_id": "poe-conversation-secret",
        "message_id": "poe-message-id",
    }
    values.update(overrides)
    return fp.QueryRequest(**values)


async def collect(bot: MemraPoeBot, request: fp.QueryRequest) -> list[fp.PartialResponse]:
    return [part async for part in bot.get_response(request)]


@pytest.mark.asyncio
async def test_translates_poe_history_and_streams_only_visible_content() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        stream = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"reasoning":"hidden"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream.encode(),
        )

    bot = MemraPoeBot(
        make_config(), transport=httpx.MockTransport(handler)
    )
    parts = await collect(bot, make_request(temperature=0.7, stop_sequences=["END"]))

    assert [part.text for part in parts] == ["Hello", " world"]
    assert captured["url"] == "http://memra.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer mk-poe-test-key"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Continue"},
    ]
    assert payload["model"] == "stepfun/step-3.7-flash"
    assert payload["max_tokens"] == 512
    assert payload["stream"] is True
    assert payload["include_reasoning"] is False
    assert payload["temperature"] == 0.7
    assert payload["stop"] == ["END"]
    assert payload["cache_salt"] == payload["session_id"]
    assert "poe-user-secret" not in str(payload["cache_salt"])
    assert "poe-conversation-secret" not in str(payload["cache_salt"])


@pytest.mark.asyncio
async def test_backend_rate_limit_is_retryable_and_does_not_leak_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(429, json={"error": "internal backend detail"})

    bot = MemraPoeBot(
        make_config(), transport=httpx.MockTransport(handler)
    )
    parts = await collect(bot, make_request())

    assert len(parts) == 1
    assert isinstance(parts[0], fp.ErrorResponse)
    assert parts[0].allow_retry is True
    assert "internal backend detail" not in parts[0].text


@pytest.mark.asyncio
async def test_context_cap_rejects_before_backend_call() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    bot = MemraPoeBot(
        make_config(max_input_chars=10),
        transport=httpx.MockTransport(handler),
    )
    request = make_request(
        [fp.ProtocolMessage(role="user", content="x" * 11)]
    )
    parts = await collect(bot, request)

    assert called is False
    assert len(parts) == 1
    assert isinstance(parts[0], fp.ErrorResponse)
    assert parts[0].allow_retry is False
    assert parts[0].error_type == "user_message_too_long"


@pytest.mark.asyncio
async def test_settings_are_free_text_only_and_non_monetized() -> None:
    bot = MemraPoeBot(make_config())
    settings = await bot.get_settings(
        fp.SettingsRequest(version="1.0", type="settings")
    )

    assert settings.allow_attachments is False
    assert settings.expand_text_attachments is False
    assert settings.enable_image_comprehension is False
    assert settings.rate_card is None
    assert settings.cost_label is None
    assert settings.parameter_controls is None


@pytest.mark.asyncio
async def test_full_poe_protocol_route_requires_access_key_and_streams() -> None:
    def backend_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    config = make_config()
    app = create_app(config, transport=httpx.MockTransport(backend_handler))
    poe_transport = httpx.ASGITransport(app=app)
    payload = make_request().model_dump()
    async with httpx.AsyncClient(
        transport=poe_transport, base_url="http://poe.test"
    ) as client:
        unauthorized = await client.post("/poe", json=payload)
        authorized = await client.post(
            "/poe",
            headers={"Authorization": f"Bearer {config.poe_access_key}"},
            json=payload,
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.headers["content-type"].startswith("text/event-stream")
    assert 'event: text\r\ndata: {"text": "Hello"}' in authorized.text
    assert "event: done" in authorized.text
