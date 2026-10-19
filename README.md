# Beaver's Choice Paper Company – Multi-Agent Sales System

Udacity *Multi-Agent Systems* final project.

## What it does

A text-in, text-out sales team for a paper company, built with at most five agents. For each
customer request it:

- checks inventory, maps the customer's wording to exact catalog items, and reorders from the
  supplier when the delivery date and cash allow;
- builds a quote at catalog prices with bulk discounts (5% at 1,000+ units, 10% at 5,000+,
  15% at 10,000+), using quote history for consistency;
- finalizes the sale in the SQLite database;
- replies to the customer with items, unit prices, discount and reason, total, delivery date, and a
  clear reason for anything that cannot be supplied.

## Framework

[smolagents](https://github.com/huggingface/smolagents) (`ToolCallingAgent` with managed agents)
using `gpt-4o-mini` through an OpenAI-compatible endpoint.

## Agents

| Agent | Responsibility |
|---|---|
| OrchestratorAgent | Delegates to the worker agents in order and writes the customer reply |
| InventoryAgent | Stock checks, catalog mapping, restock or reject decisions |
| QuotingAgent | Pricing, bulk discounts, quote history lookup |
| SalesAgent | Final stock and deadline check, records sales, internal financial health check |
| BusinessAdvisorAgent | Internal review of all transactions with recommendations (runs once at the end) |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder (it is git-ignored):

```
UDACITY_OPENAI_API_KEY=your_key_here
```

## Run

```bash
python project_starter.py
```

This processes every request in `quote_requests_sample.csv` in date order and prints an evaluation
summary.

## Outputs

- `test_results.csv` – per-request cash balance, inventory value and customer response
- `workflow_diagram.png` – agent workflow diagram (source: `workflow_diagram.mmd`)
- `report.md` – architecture explanation, evaluation results and improvement ideas
