"""Pre-flight check: credentials load and every model role is reachable.

uv run python smoke.py
"""

import asyncio
import os

import inspect_ai
from inspect_ai.model import GenerateConfig, get_model

from config import DEFAULTS, MODEL_ROLES


async def check(model_name: str) -> str:
    model = get_model(model_name)
    output = await model.generate(
        "Reply with the single word: ok",
        config=GenerateConfig(max_connections=DEFAULTS.max_connections),
    )
    return output.completion.strip() or "<empty completion>"


async def main() -> int:
    print(f"inspect_ai {inspect_ai.__version__}")

    if not os.environ.get("OPENAI_API_KEY"):
        print("FAIL  OPENAI_API_KEY not set (expected .env in the project root)")
        return 1
    print("OK    OPENAI_API_KEY loaded")

    results = await asyncio.gather(
        *(check(name) for name in MODEL_ROLES.values()),
        return_exceptions=True,
    )

    failed = 0
    for (role, name), result in zip(MODEL_ROLES.items(), results):
        if isinstance(result, Exception):
            failed += 1
            print(f"FAIL  {role:<14} {name:<24} {type(result).__name__}: {result}")
        else:
            print(f"OK    {role:<14} {name:<24} -> {result!r}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
