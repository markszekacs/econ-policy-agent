# Retrieval evaluation report

Queries: 52 | Reference point: mode=hybrid, cand_n=20, ordering=rerank_child, top_k=5 | Cells: mean [95% bootstrap CI] | Latencies: p50/p95 ms over queries

## 1. Chunking strategy

*Question: does section-aware chunking beat size-based baselines, and does the enrichment prefix contribute on its own?*

| strategy | cand recall@20 | recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| section-aware + prefix | 0.875 [0.788, 0.952] | 0.942 [0.865, 1.000] | 0.837 [0.750, 0.913] | 0.807 [0.724, 0.881] |
| section-aware, no prefix | 0.856 [0.760, 0.942] | 0.942 [0.865, 1.000] | 0.822 [0.734, 0.904] | 0.795 [0.710, 0.872] |
| recursive splitter | 0.846 [0.750, 0.933] | 0.904 [0.817, 0.971] | 0.803 [0.706, 0.891] | 0.780 [0.692, 0.862] |
| fixed windows + overlap | 0.798 [0.683, 0.904] | 0.846 [0.750, 0.942] | 0.685 [0.578, 0.790] | 0.688 [0.585, 0.790] |

> Read (section vs section_bare) for the prefix effect and (section_bare vs recursive vs fixed) for the boundary effect -- the two treatments are deliberately separable. LIMITATION: labels are synthetic-only; questions were generated from section-strategy parents, so although gold spans are quote-grounded and boundary-neutral, the question DISTRIBUTION favors section-coherent asks. Treat the section-vs-baseline margin as an upper bound.

## 2. Search mode

*Question: does the BM25 leg recover candidates the bi-encoder misses, and what does it cost in latency?*

| mode | cand recall@20 | recall@5 | MRR | nDCG@5 | search ms (p50/p95) |
|---|---|---|---|---|---|
| dense | 0.798 [0.692, 0.904] | 0.846 [0.750, 0.942] | 0.774 [0.668, 0.875] | 0.745 [0.645, 0.842] | 89 / 125 |
| keyword | 0.779 [0.673, 0.875] | 0.923 [0.846, 0.981] | 0.815 [0.722, 0.895] | 0.786 [0.695, 0.867] | 24 / 33 |
| hybrid | 0.875 [0.788, 0.952] | 0.942 [0.865, 1.000] | 0.837 [0.750, 0.913] | 0.807 [0.724, 0.881] | 113 / 158 |

> Candidate recall is the primary column here: the legs differ in what they FIND; the shared reranker handles the ordering. Hybrid latency is the sequential sum of both legs (an upper bound -- they can run in parallel in production). See section 6 for the query_type breakdown where hybrid is expected to earn its keep.

## 3. Reranking

*Question: what does the local cross-encoder reranker buy over vector/RRF order, at which granularity, and at what latency cost?*

| ordering | recall@5 | MRR | nDCG@5 | rerank ms (p50/p95) |
|---|---|---|---|---|
| rerank on children (+prefix) | 0.942 [0.865, 1.000] | 0.837 [0.750, 0.913] | 0.807 [0.724, 0.881] | 2205 / 2881 |
| rerank on parents | 0.865 [0.769, 0.942] | 0.651 [0.548, 0.755] | 0.654 [0.558, 0.744] | 7319 / 8373 |
| no rerank (RRF order) | 0.885 [0.788, 0.962] | 0.707 [0.606, 0.811] | 0.707 [0.613, 0.801] | 0 (skipped) |

> MRR is the primary column: reranking exists to move the first relevant hit up. The latency column is the price of that MRR gain -- this pair of numbers IS the reranker ROI. Note the child-vs-parent latency gap too: children are shorter documents.

## 4. Candidate budget

*Question: where does candidate recall saturate -- how many candidates are worth paying for?*

| candidates/leg | cand recall@N | recall@5 | MRR |
|---|---|---|---|
| 10 | 0.808 [0.702, 0.904] | 0.923 [0.846, 0.981] | 0.818 [0.722, 0.908] |
| 20 | 0.875 [0.788, 0.952] | 0.942 [0.865, 1.000] | 0.837 [0.750, 0.913] |
| 35 | 0.894 [0.817, 0.971] | 0.923 [0.846, 0.981] | 0.833 [0.747, 0.913] |
| 50 | 0.942 [0.885, 0.990] | 0.923 [0.846, 0.981] | 0.824 [0.734, 0.904] |

> Past the plateau, extra candidates only add rerank cost. The final recall@5 column shows whether a bigger pool ever hurts the top-5.

## 5. Context size (top_k)

*Question: how much of the gold material fits into k parents, and what does each extra parent cost in context?*

| top_k | recall@k | nDCG@k | avg context chars |
|---|---|---|---|
| 3 | 0.923 [0.846, 0.981] | 0.800 [0.713, 0.878] | 4,364 |
| 5 | 0.942 [0.865, 1.000] | 0.807 [0.724, 0.881] | 7,420 |
| 8 | 0.962 [0.904, 1.000] | 0.814 [0.734, 0.884] | 11,972 |
| 10 | 0.981 [0.942, 1.000] | 0.820 [0.745, 0.886] | 15,153 |

> Recall is monotone in k BY CONSTRUCTION -- read the MARGINAL recall gain per row against the context-chars growth: that ratio is the real decision variable (token cost and context dilution scale with chars). The downstream half of this question (does a diluted context hurt the Synthesizer) is a Tier-2 end-to-end measurement.

## 6. Breakdown by query type

*Run `python eval/score_arms.py --slice query_type` to populate this section.*

## 7. Paired comparisons (strategy=section, cand_n=20, top_k=5)

*Question: which pipeline choices make a statistically defensible difference?*

| comparison | metric | mean diff | 95% CI (paired) | verdict |
|---|---|---|---|---|
| hybrid - dense | recall | +0.0962 | [+0.0192, +0.1731] | **significant** |
| rerank - no_rerank | MRR | +0.1298 | [+0.0327, +0.2263] | **significant** |
| child - parent rerank | MRR | +0.1865 | [+0.0904, +0.2885] | **significant** |

> Paired over queries: between-query variance is removed, so these CIs are much tighter than the marginal CIs above. This table is the one to quote.
