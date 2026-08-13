# diagnose_eval.py
import json
from pathlib import Path

queries = [json.loads(l) for l in open("eval/queries.jsonl", encoding="utf-8") if l.strip()]
cache = {p.stem: p.read_text(encoding="utf-8") for p in Path("rag/.md_cache").glob("*.md")}
traces = [json.loads(l) for l in open("eval/traces/trace__section.jsonl", encoding="utf-8")]
trace_sources = {c["source"] for t in traces for c in t["candidates"]}

bad_source, bad_span = 0, 0
for q in queries:
    for g in q["gold_spans"]:
        src = g["source"]
        if src not in cache:
            print(f"{q['query_id']}: gold source '{src}' has NO canonical file")
            bad_source += 1
        elif g["char_end"] > len(cache[src]):
            print(f"{q['query_id']}: span [{g['char_start']}:{g['char_end']}] "
                  f"beyond '{src}' length {len(cache[src])}")
            bad_span += 1
        if src not in trace_sources:
            print(f"{q['query_id']}: gold source '{src}' NEVER appears in trace candidates")
print(f"\nGold sources: {sorted({g['source'] for q in queries for g in q['gold_spans']})}")
print(f"Trace candidate sources: {sorted(trace_sources)}")
print(f"Missing canonical: {bad_source}, out-of-bounds spans: {bad_span}")