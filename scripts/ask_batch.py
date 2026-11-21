"""Run questions from a text file through the copilot and print the answers. No grading."""
from __future__ import annotations

import sys
from pathlib import Path

from spendlens.copilot import REFUSAL, ask_ollama, out_of_scope, pick_ids, route, select_facts
from spendlens.copilot_facts import build_facts


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "my_questions.txt")
    if not path.exists():
        raise SystemExit(f"Create {path} with one question per line.")
    facts = build_facts()
    by_id = {f["id"]: f for f in facts}
    questions = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for n, q in enumerate(questions, start=1):
        print(f"\n=== {n}. {q}")
        if out_of_scope(q):
            print(f"[rules] {REFUSAL} (forecast)")
            continue
        routed = route(q, facts)
        if routed:
            header, ids, note = routed
            print("[rules]")
            if header:
                print(header)
        else:
            chosen = select_facts(q, facts)
            ids = pick_ids(ask_ollama(q, chosen), chosen)
            print("[model]")
            if not ids:
                print(REFUSAL)
                continue
            note = None
        for fid in ids:
            print(f"  {fid}: {by_id[fid]['text']}")
        if note:
            print(f"  Note: {note}")


if __name__ == "__main__":
    main()