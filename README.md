# langchain-agent-budget

A dollar budget a LangChain agent cannot go past. `SpendBudgetMiddleware` reserves each model
call's worst-case cost before the call and books the real cost after it. A call that does not
fit is never made, and parallel runs or deepagents subagents sharing one budget cannot spend
more than it holds together.

```bash
pip install -U langchain-agent-budget
```

```python
from agent_budget import Budget
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_agent_budget import SpendBudgetMiddleware

budget = Budget(2.00, name="support-bot")
agent = create_agent(
    ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048),
    tools=[search],
    middleware=[SpendBudgetMiddleware(budget)],
)
```

- **Worst case first**: input tokens (approximate, or your own counter) plus the call's
  `max_tokens`, priced by [llm-prices](https://github.com/Mattbusel/llm-prices) at the dearest
  input-side rate. The real cost from `usage_metadata` replaces the hold when the reply comes back.
- **Shared budgets**: one `Budget` across runs, threads and asyncio tasks; per-user or per-job
  slices with `with fleet.child("user-42", usd=0.50): agent.invoke(...)`.
- **deepagents**: give subagents `SpendBudgetMiddleware()` and they spend from the
  orchestrator's budget; `subagent_usd=` gives each subagent run its own slice.

`ModelCallLimitMiddleware` counts calls; a check after each call lets the call that crosses the
cap through. In `examples/offline_demo.py` (no API key needed) a $0.02 cap checked after each
call ends at $0.02268; the same agent with `SpendBudgetMiddleware` stops at $0.01512.

Docs: [docs/index.md](docs/index.md). Built on [agent-budget](https://github.com/Mattbusel/agent-budget)
(budgets and child scopes) and [langchain-llm-prices](https://github.com/Mattbusel/langchain-llm-prices).

MIT license. Maintained by Matthew Busel.
