# Hard dollar budgets for LangChain agents

`langchain-agent-budget` gives a `create_agent` or deepagents agent a dollar budget it cannot
go past. Before each model call, `SpendBudgetMiddleware` reserves the most that call can cost.
If the reservation does not fit, the call is not made. When the reply comes back, the hold is
swapped for the real cost and the unused part goes back.

A check that runs after each call (or before the next one, from what was spent so far) lets
the call that crosses the line through, so a run ends over its cap by up to one call. Parallel
subagents that each read the same "remaining" figure can overshoot by one call each. Reserving
first closes both gaps: the check and the reservation are one step under the budget's lock.

Budgets and child scopes come from [`agent-budget`](https://github.com/Mattbusel/agent-budget);
prices come from [`llm-prices`](https://github.com/Mattbusel/llm-prices) through
[`langchain-llm-prices`](https://github.com/Mattbusel/langchain-llm-prices).

## Overview

| Component | Class | Package | Version |
|---|---|---|---|
| middleware | `SpendBudgetMiddleware` | `langchain-agent-budget` | 0.1.0 |

## Setup

```bash
pip install -U langchain-agent-budget
```

No credentials and no network: the price table ships inside `llm-prices`.

## Usage

```python
from agent_budget import Budget
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_agent_budget import SpendBudgetMiddleware

budget = Budget(2.00, name="support-bot")  # $2 for every run of this agent

agent = create_agent(
    ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048),
    tools=[search],
    middleware=[SpendBudgetMiddleware(budget)],
)
agent.invoke({"messages": [{"role": "user", "content": "..."}]})
print(budget.spent_usd, budget.remaining_usd)
```

`SpendBudgetMiddleware(2.00)` creates the `Budget` for you.

When the next call does not fit:

- `exit_behavior="end"` (default): no call is made; the run ends with an `AIMessage` such as
  `Spend budget reached: the next model call can cost up to $0.0310 and budget 'support-bot'
  has $0.0120 left.` It carries `additional_kwargs={"lc_source": "spend_budget"}`.
- `exit_behavior="error"`: raises agent-budget's `BudgetExhausted` (or `BudgetKilled` /
  `ScopeClosed`, all subclasses of `BudgetError`).

### What is reserved

- **Output**: the call's `max_tokens` (from the request's model settings or the chat model),
  else `max_output_tokens` given to the middleware, else the model profile's maximum output.
  A lower `max_tokens` means smaller holds. If nothing caps the output, the middleware warns
  once and the hold covers the input only.
- **Input**: `count_tokens_approximately` over the system message, messages and tools, or your
  own `token_counter` (for example `model.get_num_tokens_from_messages`). It is priced at the
  dearest input-side rate the model lists (a 1-hour cache write on Anthropic), because prompt
  caching can bill input that way.
- **After the call**: the reply's `usage_metadata` is priced (cache reads and writes at their
  own rates) and booked in place of the hold. A reply with no usage books the hold.

With an exact token counter and `max_tokens` set, no call can take the budget past its limit.
With the approximate counter, the input side is an estimate, so a run can only exceed the
limit by the error in that estimate.

### Per run, per user, per job

The budget is a live object, so it can be shared however you like:

```python
from agent_budget import Budget

fleet = Budget(50.00, name="all-users")
agent = create_agent(model, tools=tools, middleware=[SpendBudgetMiddleware()])  # no budget here

with fleet.child(f"user-{user_id}", usd=0.50):  # this run gets $0.50 of the fleet budget
    agent.invoke(inputs)
```

With no budget argument, the middleware uses the budget that is current when the call is
made. The child scope's unused part goes back to the fleet when the block ends.

### Parallel runs and threads

Many runs (threads, asyncio tasks, server workers in one process) can share one `Budget`.
They are admitted one reservation at a time, so together they cannot spend more than it holds.

### deepagents subagents

```python
from agent_budget import Budget
from deepagents import create_deep_agent
from langchain_agent_budget import SpendBudgetMiddleware

budget = Budget(5.00, name="orchestrator")
researcher = {
    "name": "researcher",
    "description": "Researches one question.",
    "system_prompt": "...",
    "middleware": [SpendBudgetMiddleware()],  # spends from the orchestrator's budget
}
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    subagents=[researcher],
    middleware=[SpendBudgetMiddleware(budget, subagent_usd=1.00)],
)
```

While a `task` tool call runs, the orchestrator's budget is current, so the subagent's
`SpendBudgetMiddleware()` reserves from it. Two subagents started in parallel cannot both take
the last few cents. With `subagent_usd` (or `subagent_fraction`), each subagent run gets its
own slice, reserved before it starts and refunded when it ends, so one runaway subagent cannot
use up the others' share. `subagent_tools` names the tools that start subagents (`("task",)`
by default).

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `budget` | `None` | A `Budget` (or child scope), a dollar amount, or `None` for the current budget |
| `exit_behavior` | `"end"` | `"end"` or `"error"` |
| `max_output_tokens` | `None` | Output tokens to assume when the model sets no `max_tokens` |
| `token_counter` | `None` | Input token counter; `count_tokens_approximately` when `None` |
| `subagent_tools` | `("task",)` | Tools that run a subagent |
| `subagent_usd` | `None` | A fixed slice per subagent run |
| `subagent_fraction` | `None` | A slice of this fraction of what is left, per subagent run |
| `provider` | `None` | Provider to price with when the model does not report one |
| `unpriced` | `"refuse"` | Model with no price: `"refuse"` the call (the run ends, or `UnpricedModelError` with `exit_behavior="error"`), `"error"` (always raise), or `"warn"` (unsafe: let it through uncounted, with an `UnsafeBudgetWarning` on every call). Prices checked more than 90 days ago are refused the same way (`StalePrice`). |

## Compared with the built-in limits

`ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` count calls. `CostCapMiddleware`
(langchain-llm-prices) counts dollars per run and per thread in agent state, persisted by the
checkpointer, and checks before each call from what was already spent.
`SpendBudgetMiddleware` holds each call's worst case first, and its budget can span runs,
threads and subagents.

`examples/offline_demo.py` runs both on the same scripted agent (five tool rounds, $0.00756
per call, $0.02 cap): the after-the-fact cap makes 3 calls and spends $0.02268; the
reservation makes 2 calls and spends $0.01512.

## Maintenance

Maintained by Matthew Busel. MIT license. agent-budget and llm-prices are maintained by the
same author; llm-prices rows cite each provider's pricing page and the date it was checked.
Issues: https://github.com/Mattbusel/langchain-agent-budget/issues
