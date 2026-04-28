# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

SR2 Relay is a context-optimization proxy for LLM API calls. It wraps **LiteLLM Proxy** with custom routing, lifecycle hooks, and a YAML-driven config layer. Clients (e.g. coding assistants, agents) point their OpenAI-compatible base URL at SR2 Relay, which forwards requests to configured backends (Ollama, OpenAI, Anthropic, etc.) via LiteLLM.

## Commands

```bash
# Install / sync dependencies (uses uv, not pip)
uv sync

# Run the server (uses config/sr2-relay-config.yaml by default)
uv run sr2-relay
uv run sr2-relay path/to/config.yaml          # custom config
uv run sr2-relay --dev                         # auto-reload mode

# Tests
uv run pytest
uv run pytest tests/test_foo.py::test_bar      # single test

# Lint
uv run ruff check src/
uv run ruff format src/
```

## Architecture

```
cli.py          → argparse entry point, loads config, writes LiteLLM proxy YAML, starts uvicorn
config.py       → YAML loading, env-var resolution (${VAR} syntax), builds LiteLLM model_list config
main.py         → FastAPI app composition: imports LiteLLM proxy's app, mounts SR2 custom router, registers SR2Handler callback
sr2_handler.py  → CustomLogger subclass with lifecycle hooks (pre_call, post_call_success/failure, moderation, streaming)
api/routes.py   → SR2-specific endpoints (mounted at /sr2, currently just /sr2/health)
models/         → Pydantic config models (SR2RelayConfig, ModelSlotConfig)
```

**Request flow:** Client → FastAPI (LiteLLM proxy app) → SR2Handler.pre_call_hook → LiteLLM router → backend LLM → SR2Handler.post_call hooks → Client

**Key design decisions:**
- LiteLLM's own `app` is the root ASGI app; SR2's router is mounted onto it as a sub-router, not the other way around.
- SR2Handler sets `pass_through_all_models = True` on the LiteLLM router lazily (on first pre_call_hook), allowing any model string from clients without pre-registration.
- Config uses a single `sr2-relay-config.yaml` with three optional model slots: `model`, `fast_model`, `embedding_model`. This gets transformed into LiteLLM's `model_list` format at startup.
- Env vars in config values are resolved at load time via `${VAR_NAME}` syntax; missing vars fail fast.
- The generated LiteLLM config is written to `config/.litellm-proxy.yaml` (gitignored) and picked up via `CONFIG_FILE_PATH` env var.

## Config

`config/sr2-relay-config.yaml` — the only config file. Model slots use LiteLLM's `provider/model` naming convention. Extra fields on model slots are passed through to LiteLLM via `ConfigDict(extra="allow")`.

## Conventions

- **Python 3.12+**, managed with **uv** (not pip/poetry)
- **2-space indentation** in Python (see existing code)
- **ruff** for linting and formatting
- Source layout: `src/sr2_relay/`
- CLI entry point registered as `sr2-relay` in `[project.scripts]`
