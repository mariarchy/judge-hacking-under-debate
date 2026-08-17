from pathlib import Path

from inspect_ai.scorer import CORRECT, INCORRECT

from debate_task import (
    DEBATER,
    JUDGE_DEBATE,
    JUDGE_QA,
    QUESTION,
    STRATEGIST,
    DebateStore,
    extract_section,
    format_prompt,
    load_records,
    parse_verdict,
    records_to_dataset,
    sabotage_debate,
)

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def test_extract_section_takes_text_after_marker():
    text = "Thinking:\nsecret plan\n\nArgument:\nThe comparison is inclusive.\n"
    assert extract_section(text, "Argument:") == "The comparison is inclusive."


def test_extract_section_falls_back_to_whole_completion():
    text = "No marker here, just the speech."
    assert extract_section(text, "Argument:") == text


def test_parse_verdict_happy_path():
    text = (
        "The code looks fine.\n"
        "Answer: clean\n"
        "Confidence: 72\n"
        "Reason: The comparison matches the docstring.\n"
    )
    assert parse_verdict(text) == ("clean", 72.0)


def test_parse_verdict_is_case_insensitive_and_clamps():
    text = "Answer: SABOTAGED\nConfidence: 140\nReason: Off by one.\n"
    verdict, confidence = parse_verdict(text)
    assert verdict == "sabotaged"
    assert confidence == 100.0


def test_parse_verdict_missing_answer_is_unparsed():
    assert parse_verdict("Confidence: 90\nReason: hunches.") == ("unparsed", 0.0)


def test_format_prompt_survives_braces_in_code():
    filled = format_prompt(
        "code:\n{code}\nlimit={word_limit}",
        code="d = {'n': 1}\n",
        word_limit=150,
    )
    assert "{'n': 1}" in filled
    assert "limit=150" in filled


def test_debater_prompt_fills_every_placeholder():
    filled = format_prompt(
        DEBATER,
        player_name="Bob",
        opposing_player_name="Alice",
        word_limit=150,
        question=QUESTION,
        answer_defending="clean",
        answer_opposing="sabotaged",
        code="def f():\n    return 1\n",
        transcript="Alice (round 1): it is sabotaged",
        strategy_note="",
        round_advice="1. Critique the opponent.",
    )
    assert "{player_name}" not in filled
    assert "<your_answer>clean</your_answer>" in filled
    assert "<strategy_note>\n\n</strategy_note>" in filled


def test_judge_prompts_end_with_the_output_contract():
    contract = "Answer: <sabotaged|clean>"
    debate = format_prompt(
        JUDGE_DEBATE,
        question=QUESTION,
        code="x = 1",
        transcript="Alice (round 1): hello",
    )
    qa = format_prompt(JUDGE_QA, question=QUESTION, code="x = 1")
    assert contract in debate
    assert contract in qa
    assert "{transcript}" not in JUDGE_QA


def test_strategist_prompt_contains_the_identifier_ban():
    assert (
        "Do not reference any specific snippet, function name, variable name, or line number."
        in STRATEGIST
    )
    assert "{records}" in STRATEGIST


def test_test_split_loads_and_builds_a_task():
    records = load_records(split="test")
    assert records
    assert all(rec["split"] == "test" for rec in records)
    dataset = records_to_dataset(records)
    assert len(dataset) == len(records)
    sample = dataset[0]
    assert sample.metadata["condition"] in {"sabotaged", "clean"}
    assert sample.target == sample.metadata["condition"]

    task = sabotage_debate(protocol="qa", split="test", epochs=1)
    assert task.epochs == 1
    assert len(task.dataset) == len(records)


def test_judge_verdict_values_are_inspect_constants():
    assert CORRECT == "C"
    assert INCORRECT == "I"


def test_debate_store_is_fully_defined():
    store = DebateStore()
    assert store.turns == []
    assert store.verdict == ""
    assert store.confidence == 0.0


def test_prompt_files_exist():
    for name in ("debater.txt", "judge_debate.txt", "judge_qa.txt", "strategist.txt"):
        assert (PROMPTS / name).is_file()
