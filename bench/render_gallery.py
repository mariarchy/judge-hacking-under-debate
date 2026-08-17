"""Render snippets.json as a reviewable gallery (Markdown + HTML).

Markdown is the GitHub-native review surface. HTML is for a browser screenshot
to drop into the README.

    uv run python bench/render_gallery.py
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).parent
SNIPPETS_PATH = BENCH / "snippets.json"
MD_PATH = BENCH / "gallery.md"
HTML_PATH = BENCH / "gallery.html"


def load_snippets() -> list[dict]:
    return json.loads(SNIPPETS_PATH.read_text())


def _counts(snippets: list[dict]) -> str:
    n = len(snippets)
    by_cond = Counter(s["condition"] for s in snippets)
    parts = [f"{n} snippets"]
    if by_cond:
        parts.append(", ".join(f"{v} {k}" for k, v in sorted(by_cond.items())))
    return " · ".join(parts)


def render_markdown(snippets: list[dict]) -> str:
    lines = [
        "# Bench snippets",
        "",
        (
            f"{_counts(snippets)}. Source of truth: [`snippets.json`](snippets.json). "
            "Regenerate with `uv run python bench/render_gallery.py`."
        ),
        "",
        "| id | type | split | region | source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in snippets:
        kind = s.get("sabotage_type") or "—"
        lines.append(
            f"| `{s['id']}` | `{kind}` | {s['split']} | `{s['region']}` | `{s['source_file']}` |"
        )
    lines += ["", "---", ""]

    for s in snippets:
        lines += [
            f"## `{s['id']}`",
            "",
            (
                f"`{s['condition']}` · `{s.get('sabotage_type') or '—'}` · "
                f"{s['split']} · `{s['region']}` · `{s['source_file']}`"
            ),
            "",
        ]
        if note := s.get("sabotage_note"):
            lines += [f"> {note}", ""]
        code = s["code"].rstrip("\n")
        lines += ["```python", code, "```", ""]
    return "\n".join(lines)


def render_html(snippets: list[dict]) -> str:
    cards = []
    for s in snippets:
        kind = html.escape(s.get("sabotage_type") or "clean")
        note = (
            f'<p class="note">{html.escape(s["sabotage_note"])}</p>'
            if s.get("sabotage_note")
            else ""
        )
        cards.append(
            f"""
<article class="card" data-condition="{html.escape(s['condition'])}">
  <header>
    <h2>{html.escape(s['id'])}</h2>
    <ul class="meta">
      <li>{html.escape(s['condition'])}</li>
      <li>{kind}</li>
      <li>{html.escape(s['split'])}</li>
      <li>{html.escape(s['region'])}</li>
    </ul>
    <p class="src">{html.escape(s['source_file'])}</p>
  </header>
  {note}
  <pre><code>{html.escape(s['code'].rstrip())}</code></pre>
</article>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bench snippets</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --ink: #1b1b1b;
      --muted: #5c5852;
      --line: #d9d4c8;
      --card: #fffdf8;
      --code-bg: #1e1e1e;
      --code: #d4d4d4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0 auto;
      max-width: 1100px;
      padding: 32px 20px 64px;
      font: 15px/1.45 ui-sans-serif, system-ui, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    h1 {{ font-size: 22px; font-weight: 650; margin: 0 0 6px; }}
    .sub {{ color: var(--muted); margin: 0 0 28px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 14px 12px;
      break-inside: avoid;
    }}
    .card h2 {{
      font-size: 14px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      margin: 0 0 8px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      list-style: none;
      margin: 0;
      padding: 0;
    }}
    .meta li {{
      font-size: 11px;
      letter-spacing: 0.02em;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 1px 8px;
      color: var(--muted);
    }}
    .src {{
      margin: 8px 0 0;
      font-size: 12px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .note {{
      margin: 10px 0 0;
      font-size: 13px;
      color: var(--muted);
    }}
    pre {{
      margin: 12px 0 0;
      padding: 12px 14px;
      overflow: auto;
      background: var(--code-bg);
      color: var(--code);
      border-radius: 8px;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    code {{ font: inherit; }}
    @media print {{
      body {{ background: white; max-width: none; }}
      .card {{ box-shadow: none; }}
    }}
  </style>
</head>
<body>
  <h1>Bench snippets</h1>
  <p class="sub">{html.escape(_counts(snippets))} · from snippets.json</p>
  <div class="grid">
    {"".join(cards)}
  </div>
</body>
</html>
"""


def main() -> None:
    snippets = load_snippets()
    MD_PATH.write_text(render_markdown(snippets))
    HTML_PATH.write_text(render_html(snippets))
    print(f"Wrote {MD_PATH.relative_to(BENCH.parent)}")
    print(f"Wrote {HTML_PATH.relative_to(BENCH.parent)}")
    print("Open the HTML in a browser and screenshot; review the Markdown on GitHub.")


if __name__ == "__main__":
    main()
