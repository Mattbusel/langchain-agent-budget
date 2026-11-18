# Changelog

## 0.1.0 (2026-11-18)

First release.

- `SpendBudgetMiddleware` for `create_agent` and deepagents: reserves each model call's
  worst-case cost (approximate or exact input tokens plus `max_tokens`, priced by llm-prices
  at the dearest input-side rate) from an agent-budget `Budget` before the call, and books the
  real `usage_metadata` cost after it. `exit_behavior` `"end"` or `"error"`.
- Budgets shared across runs, threads and asyncio tasks; with no `budget` argument the
  middleware uses `current_budget()`, so `with fleet.child(...): agent.invoke(...)` gives a run
  its own slice.
- Subagents: while a `task` tool call runs the budget stays current; `subagent_usd` /
  `subagent_fraction` give each subagent run its own slice, refunded when it ends.
- Tested with deepagents 0.7: two parallel subagents against room for one call make one call.
