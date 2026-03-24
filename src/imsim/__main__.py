from __future__ import annotations

from .app import create_app
from .config import IMSimConfig


def main() -> None:
    config = IMSimConfig.from_env()
    app = create_app(config)
    app.run(debug=config.debug, host=config.host, port=config.port, threaded=True)


if __name__ == "__main__":
    main()
