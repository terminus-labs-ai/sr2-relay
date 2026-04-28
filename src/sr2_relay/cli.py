import argparse
import logging
import os

import uvicorn
import yaml
from dotenv import load_dotenv

from sr2_relay.config import build_litellm_config, load_config

load_dotenv()

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="SR2 Bridge — context optimization proxy for LLM API calls",
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
    default=None,
    action="store_true",
    help="Enable development mode (optional)",
  )
  return parser.parse_args(argv)


def run():
  """Starts the REST API server."""

  args = _parse_args()

  logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
  )

  config = load_config(args.config)
  litellm_config = build_litellm_config(config)
  litellm_config_path = "config/.litellm-proxy.yaml"
  logger.info(f"Writing litellm config: {yaml.dump(litellm_config)}")

  with open(litellm_config_path, "w") as file:
    yaml.dump(litellm_config, file, sort_keys=False, default_flow_style=False)

  os.environ["CONFIG_FILE_PATH"] = litellm_config_path
  uvicorn.run("sr2_relay.main:app", host=config.host, port=config.port, reload=args.dev)


if __name__ == "__main__":
  run()
