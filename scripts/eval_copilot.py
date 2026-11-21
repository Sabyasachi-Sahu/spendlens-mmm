"""Score the SpendLens copilot on a fixed set of questions with known answers."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from spendlens.copilot import ask_ollama, out_of_scope, pick_ids, route, select_facts
from spendlens.copilot_facts import ALL, build_facts

# (question, text that must appear in the returned facts, the only channel allowed, should be refused)
CASES = [
    ("Which channel has the best ROI?", ["paid_search 3.94"], "paid_search", False),
    ("Which channel has the worst marginal ROI?", ["rep_visits 0.61"], "rep_visits", False),
    ("Which channel is the most saturated?", ["rep_visits"], "rep_visits", False),
    ("Which channel gets the biggest budget increase?", ["paid_search +3.8"], "paid_search", False),
    ("Which channel has the biggest budget cut?", ["rep_visits -6.7"], "rep_visits", False),
    ("Which channel had the biggest ROI increase from calibration?", ["rep_visits +1.94"], "rep_visits", False),
    ("Which channel lost the most ROI after calibration?", ["patient_digital -0.53"], "patient_digital", False),
    ("What is the ROI of hcp_email?", ["hcp_email: spend"], "hcp_email", False),
    ("What happens to the patient_digital budget?", ["patient_digital: budget 24,100,000 -> 26,400,000"], "patient_digital", False),
    ("How did calibration change hcp_email?", ["hcp_email: ROI 1.85 before calibration -> 3.64 after"], "hcp_email", False),
    ("What did the lift test on hcp_email measure?", ["EXP_EMAIL_01"], "hcp_email", False),
    ("What is the marginal ROI of rep_visits?", ["rep_visits: spend"], "rep_visits", False),
    ("Does speaker_programs get more or less budget?", ["speaker_programs: budget 21,700,000 -> 17,400,000"], "speaker_programs", False),
    ("What is the total ROI of paid media?", ["All paid channels together: ROI 2.89"], ALL, False),
    ("How much of the outcome is driven by paid media?", ["paid media drives"], ALL, False),
    ("Does the total budget change?", ["Total paid budget 168,300,000 -> 168,300,000"], ALL, False),
    ("What is the total outcome lift from the budget move?", ["(+1.9%)"], ALL, False),
    ("Is this real data?", ["synthetic"], ALL, False),
    ("How is a KPI unit valued?", ["5,200"], ALL, False),
    ("How many channels have lift tests?", ["Only two channels"], ALL, False),
    ("What is the weather in Hyderabad?", [], None, True),
    ("Which channel will have the best ROI next year?", [], None, True),
    ("What is the ROI of TV?", [], None, True),
    ("What was the revenue in 2025?", [], None, True),
]


def answer(question, facts):
    """Return (how, facts returned or None if refused)."""
    if out_of_scope(question):
        return "rules", None    
    by_id = {f["id"]: f for f in facts}
    routed = route(question, facts)
    if routed:
        _, ids, _ = routed
        return "rules", [by_id[i] for i in ids]
    chosen = select_facts(question, facts)
    ids = pick_ids(ask_ollama(question, chosen), chosen)
    if not ids:
        return "model", None
    return "model", [by_id[i] for i in ids]


def grade(must, only, refuse, got):
    if refuse:
        return "OK-REFUSED" if got is None else "WRONG-ANSWERED"
    if got is None:
        return "MISSED"
    text = " ".join(f["text"] for f in got)
    if any(m not in text for m in must):
        return "WRONG"
    if only and any(f["channel"] not in (ALL, only) for f in got):
        return "WRONG"
    return "PASS"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    facts = build_facts()
    rows = []
    for q, must, only, refuse in CASES:
        how, got = answer(q, facts)
        outcome = grade(must, only, refuse, got)
        shown = " ".join(f"{f['id']}:{f['channel']}" for f in got) if got else "-"
        rows.append({"question": q, "routing": how, "outcome": outcome, "returned": shown, "refuse": refuse})
        print(f"{outcome:<15} {how:<6} {q}")
        if outcome not in ("PASS", "OK-REFUSED"):
            print(f"{'':<22}returned: {shown}")

    ans = [r for r in rows if not r["refuse"]]
    ref = [r for r in rows if r["refuse"]]
    n = lambda rs, o: sum(r["outcome"] == o for r in rs)
    print(f"\nAnswerable questions ({len(ans)}): PASS {n(ans, 'PASS')}, WRONG {n(ans, 'WRONG')}, MISSED (refused) {n(ans, 'MISSED')}")
    print(f"Should-refuse questions ({len(ref)}): refused {n(ref, 'OK-REFUSED')}, wrongly answered {n(ref, 'WRONG-ANSWERED')}")
    for how in ("rules", "model"):
        sub = [r for r in ans if r["routing"] == how]
        if sub:
            print(f"  answered by {how}: {n(sub, 'PASS')} of {len(sub)} correct")

    out = Path("reports/copilot_eval.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "routing", "outcome", "returned"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("question", "routing", "outcome", "returned")})
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()