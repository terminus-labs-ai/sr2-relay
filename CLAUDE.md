# sr2-relay

Middle tier of the SR2 stack: wraps `sr2` context management and exposes it via a Python library (`RelaySession`) and an OpenAI-compatible HTTP server.

## Stack position

```
sr2 (core context management) → sr2-relay (LLM call layer) → sr2-spectre (agent runtime)
```

## Build & test

```bash
# Install in dev mode (from sr2-relay root)
pip install -e ".[dev]"

# Run tests
pytest

# Run server
python -m sr2_relay --config configs/relay.yaml
```

## Hard rules

1. **Import only from `sr2.__init__`** — never from `sr2.*` submodules. Allowed: `SR2`, `SR2Config`, `ProcessingMode`, `ActualTokenUsage`, `ProxyContext`.
2. **Single logic path** — HTTP server calls `RelaySession.complete()`. No parallel logic.
3. **Adapter isolation** — provider translation lives in `adapters/`. Adding a provider = one new file.

## Module layout

```
src/sr2_relay/
├── __init__.py        # Public: RelaySession, RelayConfig
├── config.py          # RelayConfig pydantic model + YAML loader
├── session.py         # RelaySession — all core logic
├── server.py          # FastAPI app — thin wrapper over RelaySession
├── __main__.py        # CLI: python -m sr2_relay
├── llm.py             # litellm wrapper + callable factories
├── models.py          # Pydantic OpenAI-compatible request/response models
└── adapters/
    ├── __init__.py    # get_adapter() factory
    ├── base.py        # MessageAdapter protocol
    ├── openai.py      # Passthrough adapter
    └── anthropic.py   # Anthropic ↔ OpenAI translation
```

## sr2 API used by relay

| Symbol | Purpose |
|--------|---------|
| `SR2` | Constructed once per RelaySession |
| `SR2Config` | Constructor arg for SR2 |
| `ProcessingMode` | FULL / LIGHTWEIGHT / PASSTHROUGH |
| `ActualTokenUsage` | Token counts reported after LLM call |
| `ProxyContext` | Return type of `sr2.proxy_complete_context` |

## Testing conventions

- Mock `SR2` at the class boundary — never reach into sr2 internals.
- Mock `litellm.acompletion` at the call boundary.
- Test adapters via round-trip: `to_openai_messages → from_openai_messages`.
