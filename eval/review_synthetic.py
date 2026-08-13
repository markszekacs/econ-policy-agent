"""Interactive review of the synthetic query set. MANDATORY step.

With a synthetic-only label set, this review is the only human quality gate
before every experiment inherits the labels. Judging is much cheaper than
authoring: read the question, read the gold span it points to, and decide.

Per query you see the question and the exact gold span text, then press:
    k  keep as is
    e  edit -- keep the span (the expensive part), rewrite the question the
       way YOU would actually ask it; this recovers query realism that
       LLM-generated phrasing lacks
    d  drop (unanswerable, trivial, wrong span, duplicate topic)
    b  back one query (to fix a mis-press)
    q  quit and save progress

Drop criteria worth applying strictly:
  - The span does not actually answer the question
  - The question is trivially easy (title lookup) or unanswerably vague
  - Near-duplicate of an already-kept question
  - The question leaks the answer's exact wording

Usage:
    python eval/review_synthetic.py
Reads  eval/queries_synthetic.jsonl
Writes eval/queries.jsonl          (kept + edited records)
       eval/review_state.json      (progress, resumable)
"""

import json
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "rag" / ".md_cache"
IN_FILE = Path(__file__).parent / "queries_synthetic.jsonl"
OUT_FILE = Path(__file__).parent / "queries.jsonl"
STATE_FILE = Path(__file__).parent / "review_state.json"

_texts: dict[str, str] = {}


def _canonical(source: str) -> str:
    if source not in _texts:
        _texts[source] = (CACHE_DIR / f"{source}.md").read_text(encoding="utf-8")
    return _texts[source]


def _show(i: int, total: int, q: dict) -> None:
    print("\n" + "=" * 72)
    print(f"[{i + 1}/{total}]  {q['query_id']}  ({q['query_type']})")
    print("=" * 72)
    print(f"\nQUESTION:\n  {q['query']}\n")
    for j, span in enumerate(q["gold_spans"]):
        text = _canonical(span["source"])
        content = text[span["char_start"]:span["char_end"]]
        print(f"GOLD SPAN {j + 1} (grade {span.get('grade', 1)}, "
              f"{span['source']}, {len(content)} chars):")
        print("  " + content.replace("\n", "\n  ")[:1200])
        if len(content) > 1200:
            print("  [...span truncated for display...]")
        print()


def main() -> None:
    if not IN_FILE.exists():
        sys.exit(f"{IN_FILE} not found -- run generate_synthetic.py first.")
    queries = [json.loads(l) for l in open(IN_FILE, encoding="utf-8")
               if l.strip()]

    state = {"pos": 0, "decisions": {}}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        print(f"Resuming at {state['pos'] + 1}/{len(queries)}")

    i = state["pos"]
    while 0 <= i < len(queries):
        q = queries[i]
        _show(i, len(queries), q)
        try:
            choice = input("[k]eep / [e]dit / [d]rop / [b]ack / [q]uit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "q"

        if choice == "k":
            state["decisions"][q["query_id"]] = {"action": "keep"}
            i += 1
        elif choice == "e":
            new_q = input("  rewritten question > ").strip()
            if new_q:
                state["decisions"][q["query_id"]] = {"action": "edit",
                                                     "query": new_q}
                i += 1
            else:
                print("  (empty -- not saved, decide again)")
        elif choice == "d":
            state["decisions"][q["query_id"]] = {"action": "drop"}
            i += 1
        elif choice == "b":
            i = max(0, i - 1)
        elif choice == "q":
            break
        else:
            print("  (k / e / d / b / q)")

        state["pos"] = i
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False),
                              encoding="utf-8")

    # Write kept + edited records
    kept = []
    for q in queries:
        d = state["decisions"].get(q["query_id"])
        if d is None or d["action"] == "drop":
            continue
        rec = dict(q)
        if d["action"] == "edit":
            rec["query"] = d["query"]
            rec["origin"] = "synthetic_edited"
        kept.append(rec)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_drop = sum(1 for d in state["decisions"].values() if d["action"] == "drop")
    n_edit = sum(1 for d in state["decisions"].values() if d["action"] == "edit")
    reviewed = len(state["decisions"])
    print(f"\nReviewed {reviewed}/{len(queries)}: "
          f"{len(kept)} kept ({n_edit} edited), {n_drop} dropped.")
    print(f"-> {OUT_FILE}")
    if reviewed < len(queries):
        print("Progress saved -- rerun to continue where you left off.")
    elif n_drop == 0 and n_edit == 0:
        print("Note: 0 drops and 0 edits over the full set is a sign of a "
              "rubber-stamp review, not of a perfect generator. Consider a "
              "second, stricter pass.")


if __name__ == "__main__":
    main()