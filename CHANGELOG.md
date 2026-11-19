# Changelog

## 0.2.0 (2026-11-19)

Fail closed, with llm-agent-budget 0.2. A model the budget cannot price is no longer let
through uncounted by default: an uncounted call is how a cap stops meaning anything.

### Changed (breaking)
- `unpriced` defaults to `"refuse"`: a model llm-prices has no price for is not called, and
  the run ends with the reason (`exit_behavior="end"`) or raises `UnpricedModelError`
  (`"error"`). The old default, `"warn"`, is still available as the unsafe opt-out and now
  issues an `UnsafeBudgetWarning` on every call it lets through, not once. `"ignore"` is
  removed: an uncounted call must not be silent.
- A price checked against the provider's page more than the budget's `max_price_age_days`
  (90 by default) ago is refused like a spent budget (`StalePrice`), unless the budget's
  `Pricer` allows stale prices. `CallEstimate.checked` carries the date.
- Requires llm-agent-budget 0.2 (`>=0.2,<0.3`) and llm-prices 1.8.

### Added
- `StalePrice` and `UnsafeBudgetWarning` re-exported.

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
