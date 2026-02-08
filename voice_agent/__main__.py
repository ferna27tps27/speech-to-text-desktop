"""Allow running as: python -m voice_agent"""

import logging
import sys

from .config import Config
from .app import App

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main() -> None:
    config = Config.from_env()
    app = App(config)
    app.run()


if __name__ == "__main__":
    main()
