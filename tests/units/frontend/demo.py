"""Run the real frontend against the scripted scheduler (fixed replies; no model or GPU).

PYTHONPATH=engine:. uv run python -m tests.units.frontend.demo --text 'Hello'
"""

from __future__ import annotations

import argparse
import time

from tests.units.frontend.support import Frontend, ScriptScheduler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="你好，这是测试调度器的固定回复。")
    args = parser.parse_args()
    scheduler = ScriptScheduler(args.text)
    front = Frontend(scheduler)
    print(f"Test backend (no model): {front.url}/v1", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        front.close()
        scheduler.close()


if __name__ == "__main__":
    main()
