import argparse
import logging

import uvicorn
from dotenv import load_dotenv

from sr2_relay.config import load_config
from sr2_relay.server import create_app

load_dotenv()

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="sr2-relay — LLM format adapter and session continuity proxy",
  )
  parser.add_argument(
    "config",
    nargs="?",
    default=None,
    help="Path to YAML config file (optional, uses defaults if omitted)",
  )
  parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
  )
  parser.add_argument(
    "--dev",
    action="store_true",
    default=False,
    help="Enable development mode (auto-reload)",
  )
  return parser.parse_args(argv)


def run():
  """Start the sr2-relay HTTP server."""
  args = _parse_args()

  logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
  )

  config = load_config(args.config)
  app = create_app(config)

  logger.info(f"Starting sr2-relay on {config.host}:{config.port}")
  uvicorn.run(app, host=config.host, port=config.port, reload=args.dev)


if __name__ == "__main__":
  run()
