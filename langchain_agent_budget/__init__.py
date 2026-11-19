"""Hard dollar budgets for LangChain agents, reserved before each model call.

- `SpendBudgetMiddleware`: reserves each model call's worst-case cost from an agent-budget
  `Budget` before the call and books the real cost after it, so a run never goes past its
  budget, and parallel subagents sharing one budget cannot overspend it together.
- `CallEstimate`: what the middleware worked out before a call.

Prices come from llm-prices (through langchain-llm-prices); budgets and child scopes come from
agent-budget.
"""

from agent_budget import (
    Budget,
    BudgetError,
    BudgetExhausted,
    BudgetKilled,
    ScopeClosed,
    StalePrice,
    UnsafeBudgetWarning,
    current_budget,
)

from langchain_agent_budget.middleware import CallEstimate, SpendBudgetMiddleware

__version__ = "0.2.0"

__all__ = [
    "Budget",
    "BudgetError",
    "BudgetExhausted",
    "BudgetKilled",
    "CallEstimate",
    "ScopeClosed",
    "SpendBudgetMiddleware",
    "StalePrice",
    "UnsafeBudgetWarning",
    "__version__",
    "current_budget",
]
