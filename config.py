from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent

SABOTAGED = "sabotaged"
CLEAN = "clean"

# The judge is deliberately the weakest model here. A weak, non-reasoning judge
# supervising strong debaters is the capability asymmetry this experiment
# rests on.
# The strategist is the strongest model because it is called only NOTE_VERSIONS
# times per run, so its cost is negligible and pattern-finding is exactly what it is for.
MODEL_ROLES: dict[str, str] = {
    "judge": "openai/gpt-4.1-nano",
    "debater_sab": "openai/gpt-4.1",
    "debater_clean": "openai/gpt-4.1",
    "strategist": "openai/gpt-5",
}


@dataclass(frozen=True)
class RunConfig:
    """The configuration for a single run.

    Note: the debate budget fields are the load-bearing ones. The primary metric is
    "does judge accuracy degrade against a *fixed* debate budget", so ROUNDS and
    WORD_LIMIT must be identical across every arm and every note version.
    """

    # Debate budget fields
    rounds: int = 3
    word_limit: int = 150

    # Number of strategy-note updates. Yields note_versions + 1 checkpoints
    # (v0 is the empty note), each measured on the held-out test split.
    note_versions: int = 4

    train_epochs: int = 1
    test_epochs: int = 2
    qa_epochs: int = 3

    # Judge is pinned to greedy decoding so accuracy changes across note
    # versions are attributable to the debater, not judge sampling noise. This
    # forces the judge role to be a non-reasoning model: the gpt-5 family
    # rejects any temperature other than 1.
    judge_temperature: float = 0.0
    debater_temperature: float = 1.0
    strategist_temperature: float = 1.0

    max_connections: int = 20

    snippets_path: Path = ROOT / "bench" / "snippets.json"
    prompts_dir: Path = ROOT / "prompts"
    strategies_dir: Path = ROOT / "strategies"
    logs_dir: Path = ROOT / "logs"
    results_dir: Path = ROOT / "results"

    def note_path(self, version: int) -> Path:
        return self.strategies_dir / f"note_v{version}.md"

    def log_dir(self, arm: str) -> Path:
        """The directory to store logs for a given arm.

        Arms are the combinations of note versions and debater types (sabotaged vs. clean).
        """
        return self.logs_dir / arm


DEFAULTS = RunConfig()
