# Tier 2: agent-architecture ablation

Queries: 16 | competitors: claude-sonnet-4-6 | judge: claude-haiku-4-5-20251001 (position-swapped, consistent-verdict-only)


## Arms

- **A**: 5-agent pipeline (persona analyses + synthesizer) -- 6 LLM calls/query, avg context 18,519 chars
- **B**: single agent, single-query retrieval -- 1 LLM calls/query, avg context 8,157 chars
- **C**: single agent, multi-query (5-template union) retrieval -- 1 LLM calls/query, avg context 10,827 chars

## Pairwise results (sign test over queries)

| pair | wins | wins | ties/inconsistent | p (sign test) |
|---|---|---|---|---|
| A vs C | A: 15 | C: 0 | 1 | 0.000 **significant** |
| C vs B | C: 5 | B: 2 | 9 | 0.453 |
| A vs B | A: 15 | B: 0 | 1 | 0.000 **significant** |

## Reading guide

- **A vs C** isolates THINKING diversity (both see multi-query retrieval; only A has 5 separate persona analyses).
- **C vs B** isolates RETRIEVAL diversity (same single analyst; different candidate pools).
- A ~ C with A at 6x the LLM calls means the agent layer's value is not in the answers -- the multi-query retrieval alone captures it.

Limitations: competitors and judge share a model family (symmetric across arms, but absolute quality is not validated); agent prompts approximate the production personas; the Critic stage is omitted from all arms.
