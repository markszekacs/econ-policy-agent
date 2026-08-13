"""Post-hoc scoring: load a RunLog, score each agent, write judge_scores back to file."""

import json
from pathlib import Path

from eval.judges.llm_judge import score_agent_quality

AGENT_NAMES = [
    "macroeconomist",
    "labor_economist",
    "trade_unionist",
    "institutional",
    "fiscal_expert",
]


def score_run_log(log_path: Path) -> dict:
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)

    query = log["query"]
    judge_scores: dict = {}

    for agent_name in AGENT_NAMES:
        agent_log = log.get(agent_name, {})
        if agent_log.get("status") != "success":
            continue
        response = agent_log.get("response")
        if not response:
            continue

        try:
            scores = score_agent_quality(
                query=query,
                agent_name=agent_name,
                agent_response=response,
                evidence_strength=agent_log.get("evidence_strength"),
            )
            if scores is not None:
                judge_scores[agent_name] = scores.model_dump()
            else:
                judge_scores[agent_name] = {"error": "scoring returned no result"}
        except Exception as e:
            judge_scores[agent_name] = {"error": str(e)}

    log["judge_scores"] = judge_scores

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    return judge_scores


def score_all_logs(logs_dir: Path) -> None:
    log_files = list(logs_dir.rglob("*.json"))
    for path in log_files:
        try:
            with open(path) as f:
                log = json.load(f)
            if log.get("judge_scores") is not None:
                continue  # already scored
            print(f"Scoring {path.name} ...")
            score_run_log(path)
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    from pathlib import Path
    score_all_logs(Path("logs"))
