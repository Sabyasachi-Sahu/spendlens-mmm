"""SpendLens insight copilot (extractive).

Ranking questions (best/worst ROI or marginal ROI) are answered by rules in code.
For other questions a local Ollama model only chooses which numbered facts answer the question.
Either way the answer is the facts, word for word, so a number or meaning cannot be changed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

from spendlens.copilot_facts import ALL, build_facts

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"
MAX_FACTS = 26
MAX_PICK = 3
REFUSAL = "I can't answer that from the exported results."

ALIASES = {
    "rep_visits": ["rep_visits", "rep visits", "rep visit", "reps"],
    "speaker_programs": ["speaker_programs", "speaker program", "speaker"],
    "hcp_email": ["hcp_email", "hcp email", "email"],
    "programmatic_display": ["programmatic_display", "programmatic", "display"],
    "paid_search": ["paid_search", "paid search", "search"],
    "paid_social": ["paid_social", "paid social", "social"],
    "medical_journals": ["medical_journals", "medical journal", "journal"],
    "patient_digital": ["patient_digital", "patient digital", "patient"],
}
TOPIC_WORDS = {
    "budget": ["budget", "reallocat", "shift", "move", "optimi", "increase", "decrease", "cut", "invest", "spend more", "spend less"],
    "calibration": ["calibrat", "experiment", "lift test", "before and after", "uncalibrated"],
    "performance": ["roi", "marginal", "mroi", "return", "saturat", "perform", "best", "worst", "rank", "top", "efficien", "pay"],
    "caveat": ["limit", "caveat", "synthetic", "trust", "uncertain", "confiden", "reliab", "assum", "kpi", "valued"],
}
TOPIC_KINDS = {
    "budget": {"budget", "budget_overview", "caveat"},
    "calibration": {"calibration", "calibration_overview", "experiment", "caveat"},
    "performance": {"channel", "overview"},
    "caveat": {"caveat"},
}
RANK_PREFIX = {
    "mroi": "Marginal ROI at current spend, highest first",
    "roi": "ROI, highest first",
}
RANK_LABEL = {"mroi": "marginal ROI", "roi": "ROI"}
BEST = ["best", "top", "highest", "largest", "biggest", "strongest", "most"]
WORST = ["worst", "lowest", "weakest", "smallest", "least", "bottom"]

SYSTEM = (
    "You select facts. You are given numbered facts and a question.\n"
    'Reply with JSON only, in this form: {"fact_ids": ["F001", "F002"]}\n'
    "List the 1 to 3 facts that best answer the question.\n"
    'If no fact answers the question, reply {"fact_ids": []}.\n'
    "Write nothing except the JSON."
)


def _has(q, words):
    return any(re.search(rf"\b{w}\b", q) for w in words)

def route_budget(q, facts):
    """Rules for biggest budget increase or cut questions, ranked in million INR."""
    cut = re.search(r"\b(cut|cuts|reduc\w*|decrease\w*|lose|loses|shrink\w*|drop\w*)\b", q)
    up = re.search(r"\b(increase\w*|gain\w*|grow\w*|boost\w*|extra)\b", q)
    if re.search(r"\b(smallest|lowest|least)\b", q) or not (cut or up) or (cut and up):
        return None
    rank_fact = next((f for f in facts if f["text"].startswith("Budget change in million INR")), None)
    if rank_fact is None:
        return None
    body = rank_fact["text"].split(":", 1)[1]
    pairs = re.findall(r"([a-z_]+) ([+-]\d+(?:\.\d+)?)", body)
    if len(pairs) < 2:
        return None
    idx = len(pairs) - 1 if cut else 0
    neighbour = idx - 1 if cut else 1
    channel, value = pairs[idx]
    note = "Ranked by amount in million INR. By percent the order can differ slightly, because budgets are rounded to 100,000."
    if pairs[neighbour][1] == value:
        note = f"{pairs[neighbour][0]} has the same change ({value}). " + note
    ids = [rank_fact["id"]]
    hit = next((f for f in facts if f["kind"] == "budget" and f["channel"] == channel), None)
    if hit:
        ids.append(hit["id"])
    header = f"Largest budget {'cut' if cut else 'increase'}: {channel} ({value} million INR)."
    return header, ids, note
def route_calibration(q, facts):
    """Rules for which channel calibration changed the most (or least)."""
    if not re.search(r"\bcalibrat\w*\b", q):
        return None
    if not (re.search(r"\b(which|what)\b", q) and re.search(r"\bchannels?\b", q)):
        return None
    down = re.search(r"\b(decrease\w*|drop\w*|fell|fall\w*|lost|lose\w*|reduc\w*|smallest|lowest|least)\b", q)
    up = re.search(r"\b(biggest|largest|most|highest|greatest|increase\w*|rose|gain\w*)\b", q)
    if not (down or up):
        return None
    rank_fact = next((f for f in facts if f["text"].startswith("ROI change from calibration")), None)
    if rank_fact is None:
        return None
    body = rank_fact["text"].split(":", 1)[1]
    pairs = re.findall(r"([a-z_]+) ([+-]\d+(?:\.\d+)?)", body)
    if len(pairs) < 2:
        return None
    idx = len(pairs) - 1 if down else 0
    neighbour = idx - 1 if down else 1
    channel, value = pairs[idx]
    note = None
    if pairs[neighbour][1] == value:
        note = f"{pairs[neighbour][0]} has the same change ({value}) at two decimals."
    ids = [rank_fact["id"]]
    hit = next((f for f in facts if f["kind"] == "calibration" and f["channel"] == channel), None)
    if hit:
        ids.append(hit["id"])
    header = f"Largest ROI {'decrease' if down else 'increase'} from calibration: {channel} ({value})."
    return header, ids, note
FORECAST = re.compile(
    r"\b(next (?:\d+ )?(?:years?|quarters?|months?|weeks?|days?)|over the next|"
    r"coming (?:years?|quarters?|months?|weeks?)|forecast\w*|predict\w*|projection\w*|"
    r"in the future|20(?:2[7-9]|[3-9]\d))\b"
)


def out_of_scope(q):
    """The results describe the past. They are not a forecast."""
    return bool(FORECAST.search(q.lower()))

def route_total(q, facts):
    """Rules for the ROI of all paid channels together."""
    if not re.search(r"\b(total|overall|combined|together|all paid|paid media|all channels)\b", q):
        return None
    if not re.search(r"\b(roi|return|returns)\b", q):
        return None
    if re.search(r"\b(marginal|mroi|calibrat\w*|budget)\b", q):
        return None
    hit = next((f for f in facts if f["text"].startswith("All paid channels together")), None)
    if hit is None:
        return None
    return None, [hit["id"]], None

def route(question, facts):
    """Rules for best/worst ROI or marginal ROI questions. Returns (header, ids, note) or None."""
    q = question.lower()
    if re.search(r"\b(calibrat\w*|experiment\w*|lift|uncalibrated|before|after)\b", q):
        return route_calibration(q, facts)
    if not (_has(q, BEST) or _has(q, WORST)):
        hit = route_total(q, facts)
        if hit:
            return hit    
    if not re.search(r"\b(roi|mroi|marginal|return|returns|saturat\w*)\b", q):
        hit = route_budget(q, facts)
        if hit:
            return hit
    best, worst = _has(q, BEST), _has(q, WORST)
    if best == worst:  # neither, or both: not a clear ranking question
        return None
    sat = bool(re.search(r"\bsaturat\w*\b", q))
    if sat:  # "most saturated" means the LOWEST marginal ROI
        best, worst = worst, best    
    if re.search(r"\b(marginal|mroi|saturat\w*)\b", q):
        metric = "mroi"
    elif re.search(r"\b(roi|return|returns)\b", q):
        metric = "roi"
    else:
        return None
    rank_fact = next((f for f in facts if f["text"].startswith(RANK_PREFIX[metric])), None)
    if rank_fact is None:
        return None
    body = rank_fact["text"].split(":", 1)[1]
    pairs = re.findall(r"([a-z_]+) (-?\d+(?:\.\d+)?)", body)
    if len(pairs) < 2:
        return None
    idx = 0 if best else len(pairs) - 1
    neighbour = 1 if best else len(pairs) - 2
    channel, value = pairs[idx]
    note = None
    if pairs[neighbour][1] == value:
        note = (f"{pairs[neighbour][0]} shows the same value ({value}) at two decimals; "
                "the order comes from the unrounded figures.")

    ids = [rank_fact["id"]]
    for kind, trigger in (
        ("channel", True),
        ("budget", bool(re.search(r"\b(budget|reallocat\w*|spend)\b", q))),
        ("calibration", "calibrat" in q),
    ):
        if trigger:
            hit = next((f for f in facts if f["kind"] == kind and f["channel"] == channel), None)
            if hit:
                ids.append(hit["id"])
    header = f"{'Best' if best else 'Worst'} by {RANK_LABEL[metric]}: {channel} ({value})."
    if sat:
        note = "Most saturated = lowest marginal ROI at current spend." + (f" {note}" if note else "")    
    return header, ids, note


def select_facts(question, facts):
    q = question.lower()
    channels = {
        c for c, names in ALIASES.items()
        if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names)
    }
    kinds = set()
    for topic, words in TOPIC_WORDS.items():
        if any(w in q for w in words):
            kinds |= TOPIC_KINDS[topic]
    if not kinds:
        kinds = {"channel", "overview", "budget_overview", "caveat"}
    chosen = [
        f for f in facts
        if f["kind"] in kinds and (not channels or f["channel"] in channels or f["channel"] == ALL)
    ]
    return chosen[:MAX_FACTS]


def ask_ollama(question, chosen):
    facts_text = "\n".join(f"{f['id']}: {f['text']}" for f in chosen)
    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "keep_alive": "3m",
        "options": {"temperature": 0, "num_ctx": 3072, "num_predict": 80},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Facts:\n{facts_text}\n\nQuestion: {question}"},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()
    except urllib.error.URLError:
        raise SystemExit("Cannot reach Ollama at localhost:11434. Open the Ollama app and try again.")


def pick_ids(raw, chosen):
    """Return the chosen fact IDs, or None if the reply is not a valid selection."""
    try:
        ids = json.loads(raw).get("fact_ids", [])
    except (ValueError, AttributeError):
        return None
    if not isinstance(ids, list):
        return None
    valid = {f["id"] for f in chosen}
    ids = [i for i in ids if isinstance(i, str)]
    if any(i not in valid for i in ids):
        return None
    seen = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return seen[:MAX_PICK]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+")
    p.add_argument("--debug", action="store_true", help="show how the facts were chosen")
    args = p.parse_args()
    question = " ".join(args.question)
    if out_of_scope(question):
        print(f"\n{REFUSAL}  (the results describe the past; they are not a forecast)")
        return    

    facts = build_facts()
    by_id = {f["id"]: f for f in facts}
    header = note = None

    routed = route(question, facts)
    if routed:
        header, ids, note = routed
        if args.debug:
            print("[routing: rules]")
    else:
        chosen = select_facts(question, facts)
        if args.debug:
            print(f"[routing: model, {len(chosen)} facts sent to {MODEL}]")
        raw = ask_ollama(question, chosen)
        if args.debug:
            print(f"Model reply: {raw}")
        ids = pick_ids(raw, chosen)
        if not ids:
            why = "no valid fact chosen" if ids is not None else "invalid selection"
            print(f"\n{REFUSAL}  ({why})")
            return

    print()
    if header:
        print(header)
    for fid in ids:
        print(f"- {by_id[fid]['text']} [{fid}]")
    if note:
        print(f"Note: {note}")
    print("\nSources:")
    for fid in ids:
        print(f"  {fid}  {by_id[fid]['source']}")


if __name__ == "__main__":
    main()