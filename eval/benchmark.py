"""Benchmark query set for eval experiments."""

BENCHMARK_QUERIES = [
    {
        "id": "us_ubi",
        "query": "Should the United States implement Universal Basic Income?",
        "category": "social_policy",
        "region": "US",
        "expected_disagreement": "high",
    },
    {
        "id": "us_tariffs",
        "query": "What are the economic effects of Trump's tariff policy on the US economy?",
        "category": "trade_policy",
        "region": "US",
        "expected_disagreement": "high",
    },
    {
        "id": "us_minimum_wage",
        "query": "Should the US federal minimum wage be raised to $20 per hour?",
        "category": "labor_policy",
        "region": "US",
        "expected_disagreement": "medium",
    },
    {
        "id": "us_deficit",
        "query": "Is the US fiscal deficit a threat to long-term economic stability?",
        "category": "fiscal_policy",
        "region": "US",
        "expected_disagreement": "medium",
    },
    {
        "id": "us_wealth_tax",
        "query": "Should the United States introduce a federal wealth tax?",
        "category": "tax_policy",
        "region": "US",
        "expected_disagreement": "high",
    },
    {
        "id": "hu_ubi",
        "query": "Should Hungary implement Universal Basic Income?",
        "category": "social_policy",
        "region": "HU",
        "expected_disagreement": "high",
    },
    {
        "id": "eu_carbon_tax",
        "query": "Should the EU introduce a universal carbon tax?",
        "category": "climate_policy",
        "region": "EU",
        "expected_disagreement": "medium",
    },
    {
        "id": "eu_austerity",
        "query": "Is fiscal austerity the right response to high public debt in the EU?",
        "category": "fiscal_policy",
        "region": "EU",
        "expected_disagreement": "high",
    },
    {
        "id": "eu_ai_regulation",
        "query": "Should AI development be more strictly regulated in Europe?",
        "category": "tech_policy",
        "region": "EU",
        "expected_disagreement": "medium",
    },
    {
        "id": "eu_four_day_week",
        "query": "What would be the economic impact of a four-day work week in Europe?",
        "category": "labor_policy",
        "region": "EU",
        "expected_disagreement": "medium",
    },
]

BENCHMARK_BY_ID = {q["id"]: q for q in BENCHMARK_QUERIES}
