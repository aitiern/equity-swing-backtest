"""Alpaca configuration, loaded from the environment (or a git-ignored .env).

A hard guard refuses to run against a live account: this project trades paper only,
because the research found no demonstrated edge. Removing that guard is an explicit,
deliberate act — not something that can happen by a stray env var.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class AlpacaConfig:
    api_key: str
    secret_key: str
    paper: bool = True
    sector: str = "tech"
    strategy: str = "donchian"

    @classmethod
    def from_env(cls) -> AlpacaConfig:
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"
        if not key or not secret:
            raise RuntimeError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "(copy .env.example to .env and fill them in)."
            )
        if not paper:
            raise RuntimeError(
                "Refusing to run: ALPACA_PAPER is false. This project is PAPER-ONLY by "
                "design — there is no demonstrated edge to risk real capital on."
            )
        return cls(
            api_key=key,
            secret_key=secret,
            paper=True,
            sector=os.getenv("TRADE_SECTOR", "tech"),
            strategy=os.getenv("TRADE_STRATEGY", "donchian"),
        )
