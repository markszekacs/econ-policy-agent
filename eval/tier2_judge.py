"""Tier 2 judge + report: pairwise answer comparison with bias controls.

Protocol (each control exists for a documented bias):
  - PAIRWISE comparison, not absolute scoring: judges are unreliable on
    absolute scales but decent at "which of these two is better".
  - POSITION SWAP: every pair is judged twice with the answers in swapped
    order; judges have a known first-position bias. Only a CONSISTENT
    verdict (same winner in both orders) counts as a win -- inconsistent
    verdicts count as ties.
  - Cheap judge model, different from... note: competitors and judge are
    both Claude-family (self-preference bias exists but is SYMMETRIC across
    arms here, since all arms use the same competitor model -- record it as
    a limitation, it does not invalidate arm-vs-arm comparison).
  - Sign test over queries for each arm pair (exact binomial, ties dropped).

Runs on the saved answers only -- re-judging with a different model or
rubric never requires re-generating answers.

Usage:
    py -3.12 eval\ tier2_judge.py
Writes eval/tier2/judgments.jsonl and eval/tier2/tier2_report.md
"""

import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent / "tier2"
ANSWERS_FILE = OUT_DIR / "answers.jsonl"
JUDGMENTS_FILE = OUT_DIR / "judgments.jsonl"
REPORT_FILE = OUT_DIR / "tier2_report.md"

JUDGE_MODEL = "claude-haiku-4-5-20251001"
PAIRS = [("A", "C"), ("C", "B"), ("A", "B")]

ARM_LABELS = {
    "A": "5-agent pipeline (persona analyses + synthesizer)",
    "B": "single agent, single-query retrieval",
    "C": "single agent, multi-query (5-template union) retrieval",
}

JUDGE_TEMPLATE = """You are evaluating two anonymous economic policy analyses
answering the same question. Judge ONLY the text in front of you.

Question: {query}

--- ANSWER 1 ---
{answer_1}

--- ANSWER 2 ---
{answer_2}

Compare them on, in order of importance:
1. Coverage: does the answer address the economically important dimensions
   (effects, risks, distributional impact, feasibility, fiscal cost)?
2. Groundedness: are claims tied to cited sources [S#] rather than asserted?
3. Internal consistency and clarity of the recommendation.
4. Decision-usefulness for a policy maker.

Length alone is NOT quality. Return ONLY this JSON:
{{"winner": "1" | "2" | "tie", "reason": "<one sentence>"}}"""


def call_judge(prompt: str, retries: int = 4) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=JUDGE_MODEL, max_tokens=300,
                messages=[{"role": "user", "content": prompt}])
            raw = msg.content[0].text.strip()
            raw = raw.removeprefix("```json").removesuffix("```").strip()
            return json.loads(raw)
        except Exception as exc:
            if attempt == retries - 1:
                return {"winner": "tie", "reason": f"judge failed: {exc}"}
            time.sleep(2 ** (attempt + 1))


def sign_test_p(wins_a: int, wins_b: int) -> float:
    """Two-sided exact binomial sign test, ties excluded."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    answers: dict[tuple[str, str], dict] = {}
    for line in open(ANSWERS_FILE, encoding="utf-8"):
        r = json.loads(line)
        answers[(r["query_id"], r["arm"])] = r
    query_ids = sorted({qid for qid, _ in answers})
    queries = {qid: answers[(qid, "A")]["query"] for qid in query_ids
               if (qid, "A") in answers}

    done = set()
    if JUDGMENTS_FILE.exists():
        for line in open(JUDGMENTS_FILE, encoding="utf-8"):
            j = json.loads(line)
            done.add((j["query_id"], j["arm_x"], j["arm_y"], j["order"]))

    # Judge every pair in both orders (resumable)
    for qid in query_ids:
        for arm_x, arm_y in PAIRS:
            if (qid, arm_x) not in answers or (qid, arm_y) not in answers:
                continue
            for order, (first, second) in enumerate(
                    [(arm_x, arm_y), (arm_y, arm_x)]):
                if (qid, arm_x, arm_y, order) in done:
                    continue
                verdict = call_judge(JUDGE_TEMPLATE.format(
                    query=queries[qid],
                    answer_1=answers[(qid, first)]["answer"],
                    answer_2=answers[(qid, second)]["answer"]))
                winner_arm = ("tie" if verdict.get("winner") == "tie"
                              else first if verdict.get("winner") == "1"
                              else second)
                rec = {"query_id": qid, "arm_x": arm_x, "arm_y": arm_y,
                       "order": order, "first_shown": first,
                       "winner_arm": winner_arm,
                       "reason": verdict.get("reason", "")}
                with open(JUDGMENTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{qid} {arm_x}v{arm_y} order{order}: {winner_arm}")

    # Aggregate: consistent verdicts only
    judgments = [json.loads(l) for l in open(JUDGMENTS_FILE, encoding="utf-8")]
    by_key: dict[tuple, list] = {}
    for j in judgments:
        by_key.setdefault((j["query_id"], j["arm_x"], j["arm_y"]), []).append(j)

    lines = ["# Tier 2: agent-architecture ablation\n",
             f"Queries: {len(query_ids)} | competitors: "
             f"{next(iter(answers.values()))['model']} | judge: {JUDGE_MODEL} "
             f"(position-swapped, consistent-verdict-only)\n",
             "\n## Arms\n"]
    for arm, label in ARM_LABELS.items():
        n_calls = [a["n_llm_calls"] for a in answers.values()
                   if a["arm"] == arm]
        ctx = [a["context_chars"] for a in answers.values() if a["arm"] == arm]
        lines.append(f"- **{arm}**: {label} -- {n_calls[0] if n_calls else '?'} "
                     f"LLM calls/query, avg context "
                     f"{sum(ctx) // max(1, len(ctx)):,} chars")

    lines.append("\n## Pairwise results (sign test over queries)\n")
    lines.append("| pair | wins | wins | ties/inconsistent | p (sign test) |")
    lines.append("|---|---|---|---|---|")
    for arm_x, arm_y in PAIRS:
        wx = wy = t = 0
        for (qid, ax, ay), pair in by_key.items():
            if (ax, ay) != (arm_x, arm_y) or len(pair) < 2:
                continue
            winners = {p["winner_arm"] for p in pair}
            if winners == {arm_x}:
                wx += 1
            elif winners == {arm_y}:
                wy += 1
            else:
                t += 1
        p = sign_test_p(wx, wy)
        sig = " **significant**" if p < 0.05 else ""
        lines.append(f"| {arm_x} vs {arm_y} | {arm_x}: {wx} | {arm_y}: {wy} "
                     f"| {t} | {p:.3f}{sig} |")

    lines.append(
        "\n## Reading guide\n\n"
        "- **A vs C** isolates THINKING diversity (both see multi-query "
        "retrieval; only A has 5 separate persona analyses).\n"
        "- **C vs B** isolates RETRIEVAL diversity (same single analyst; "
        "different candidate pools).\n"
        "- A ~ C with A at 6x the LLM calls means the agent layer's value "
        "is not in the answers -- the multi-query retrieval alone captures "
        "it.\n\n"
        "Limitations: competitors and judge share a model family "
        "(symmetric across arms, but absolute quality is not validated); "
        "agent prompts approximate the production personas; the Critic "
        "stage is omitted from all arms.\n")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nReport -> {REPORT_FILE}")


if __name__ == "__main__":
    main()