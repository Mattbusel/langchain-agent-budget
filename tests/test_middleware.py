from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from agent_budget import Budget, BudgetExhausted, StalePrice, UnsafeBudgetWarning
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_llm_prices import CostCapMiddleware, UnpricedModelError

from langchain_agent_budget import SpendBudgetMiddleware

from .conftest import IN_RATE, OUT_RATE, ScriptedChat, reply

CALL_COST = 20 * IN_RATE + 500 * OUT_RATE  # one scripted call: $0.00756


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"result {q}"


def run(agent: Any) -> list[Any]:
    return agent.invoke({"messages": [HumanMessage("go")]})["messages"]


def test_books_the_real_cost_and_returns_the_rest(tool_loop: Any) -> None:
    budget = Budget(1.00)
    model = ScriptedChat(respond=tool_loop(2))
    agent = create_agent(model, tools=[lookup], middleware=[SpendBudgetMiddleware(budget)])
    messages = run(agent)
    assert messages[-1].content == "answer"
    assert model.calls == 3
    assert budget.spent_usd == pytest.approx(3 * CALL_COST)
    assert budget.reserved_usd == 0


def test_no_overshoot_where_a_post_call_check_overshoots(tool_loop: Any) -> None:
    """The gap from langchain-ai/langchain#38842: a check after each call lets the call that
    crosses the cap through. Reserving the worst case first refuses it."""
    cap = 0.02  # room for two calls of $0.00756, not three

    after_model = ScriptedChat(respond=tool_loop(5))
    capped = create_agent(
        after_model, tools=[lookup], middleware=[CostCapMiddleware(max_usd_per_run=cap)]
    )
    run(capped)
    assert after_model.calls == 3  # the third call started under the cap and crossed it
    overshoot_spend = 3 * CALL_COST
    assert overshoot_spend > cap

    budget = Budget(cap)
    reserved = ScriptedChat(respond=tool_loop(5))
    agent = create_agent(reserved, tools=[lookup], middleware=[SpendBudgetMiddleware(budget)])
    messages = run(agent)
    assert reserved.calls == 2
    assert budget.spent_usd == pytest.approx(2 * CALL_COST)
    assert budget.spent_usd <= cap
    assert messages[-1].additional_kwargs["lc_source"] == "spend_budget"
    assert "Spend budget reached" in messages[-1].content


def test_error_behavior_raises_before_the_call(tool_loop: Any) -> None:
    model = ScriptedChat(respond=tool_loop(0))
    agent = create_agent(model, middleware=[SpendBudgetMiddleware(0.005, exit_behavior="error")])
    with pytest.raises(BudgetExhausted):
        run(agent)
    assert model.calls == 0


def test_hold_follows_max_tokens_from_model_settings() -> None:
    model = ScriptedChat(respond=lambda m: reply(), max_tokens=None)
    mw = SpendBudgetMiddleware(1.0)
    from langchain.agents.middleware.types import ModelRequest

    request = ModelRequest(
        model=model, messages=[HumanMessage("hi")], model_settings={"max_tokens": 2000}
    )
    estimate = mw.estimate(request)
    assert estimate.max_output_tokens == 2000
    assert estimate.provider == "anthropic"
    # input priced at the 1-hour cache-write rate ($6/M), the dearest input-side rate
    expected = estimate.input_tokens * 6.0 / 1_000_000 + 2000 * OUT_RATE
    assert estimate.cost_usd == pytest.approx(expected)


def test_falls_back_to_middleware_then_profile_output_cap() -> None:
    from langchain.agents.middleware.types import ModelRequest

    model = ScriptedChat(respond=lambda m: reply(), max_tokens=None)
    request = ModelRequest(model=model, messages=[HumanMessage("hi")])
    assert (
        SpendBudgetMiddleware(1.0, max_output_tokens=300).estimate(request).max_output_tokens == 300
    )
    profiled = ScriptedChat(
        respond=lambda m: reply(), max_tokens=None, profile={"max_output_tokens": 64000}
    )
    request = ModelRequest(model=profiled, messages=[HumanMessage("hi")])
    assert SpendBudgetMiddleware(1.0).estimate(request).max_output_tokens == 64000


def test_threads_sharing_a_budget_cannot_overspend_it() -> None:
    """Eight runs at once against room for three calls: exactly three calls are made."""
    budget = Budget(3 * CALL_COST + 0.001)
    model = ScriptedChat(respond=lambda m: reply("answer"), delay=0.05)
    agent = create_agent(model, middleware=[SpendBudgetMiddleware(budget)])
    start = threading.Barrier(8)

    def one(_: int) -> str:
        start.wait()
        return str(run(agent)[-1].content)

    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(one, range(8)))
    assert model.calls == 3
    assert answers.count("answer") == 3
    assert budget.spent_usd == pytest.approx(3 * CALL_COST)
    assert budget.spent_usd <= budget.limit_usd


def test_budget_from_context_per_run(tool_loop: Any) -> None:
    fleet = Budget(1.00, name="fleet")
    model = ScriptedChat(respond=tool_loop(1))
    agent = create_agent(model, tools=[lookup], middleware=[SpendBudgetMiddleware()])
    with fleet.child("job-1", usd=0.10) as job:
        run(agent)
        assert job.spent_usd == pytest.approx(2 * CALL_COST)
    assert fleet.spent_usd == pytest.approx(2 * CALL_COST)
    assert fleet.reserved_usd == 0


def test_no_budget_anywhere_is_an_error() -> None:
    agent = create_agent(
        ScriptedChat(respond=lambda m: reply()), middleware=[SpendBudgetMiddleware()]
    )
    with pytest.raises(ValueError, match="no budget"):
        run(agent)


def test_missing_usage_books_the_worst_case() -> None:
    budget = Budget(1.00)
    model = ScriptedChat(respond=lambda m: reply(usage=False))
    agent = create_agent(model, middleware=[SpendBudgetMiddleware(budget)])
    with pytest.warns(UserWarning, match="no usage_metadata"):
        run(agent)
    assert budget.spent_usd > 500 * OUT_RATE  # the hold, not $0


def test_unpriced_model_policy() -> None:
    model = ScriptedChat(
        respond=lambda m: reply(served_model="my-finetune"), model_name="my-finetune"
    )
    budget = Budget(1.00)
    # The default refuses: no call is made, the run ends with the reason.
    agent = create_agent(model, middleware=[SpendBudgetMiddleware(budget)])
    result = run(agent)
    assert "refusing to guess" in result[-1].content
    assert model.calls == 0
    strict = create_agent(model, middleware=[SpendBudgetMiddleware(budget, unpriced="error")])
    with pytest.raises(UnpricedModelError):
        run(strict)
    # The unsafe opt-out warns on every call it lets through.
    loose = create_agent(model, middleware=[SpendBudgetMiddleware(budget, unpriced="warn")])
    with pytest.warns(UnsafeBudgetWarning, match="not counted"):
        run(loose)
    with pytest.warns(UnsafeBudgetWarning, match="not counted"):
        run(loose)
    assert budget.spent_usd == 0
    with pytest.raises(ValueError, match="ignore"):
        SpendBudgetMiddleware(budget, unpriced="ignore")  # type: ignore[arg-type]


def test_stale_prices_are_refused() -> None:
    from agent_budget import Pricer

    model = ScriptedChat(respond=lambda m: reply())
    future = Budget(1.00, pricer=Pricer(today="2099-01-01"))
    agent = create_agent(model, middleware=[SpendBudgetMiddleware(future)])
    result = run(agent)
    assert "last checked" in result[-1].content
    assert future.spent_usd == 0
    strict = create_agent(model, middleware=[SpendBudgetMiddleware(future, exit_behavior="error")])
    with pytest.raises(StalePrice):
        run(strict)


def test_failed_call_releases_its_hold() -> None:
    def boom(messages: Any) -> AIMessage:
        raise RuntimeError("provider down")

    budget = Budget(1.00)
    agent = create_agent(ScriptedChat(respond=boom), middleware=[SpendBudgetMiddleware(budget)])
    with pytest.raises(RuntimeError):
        run(agent)
    assert budget.spent_usd == 0
    assert budget.reserved_usd == 0


async def test_async_run(tool_loop: Any) -> None:
    budget = Budget(0.02)
    model = ScriptedChat(respond=tool_loop(5))
    agent = create_agent(model, tools=[lookup], middleware=[SpendBudgetMiddleware(budget)])
    result = await agent.ainvoke({"messages": [HumanMessage("go")]})
    assert model.calls == 2
    assert budget.spent_usd <= 0.02
    assert "Spend budget reached" in result["messages"][-1].content


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        SpendBudgetMiddleware(1.0, exit_behavior="stop")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SpendBudgetMiddleware(1.0, subagent_usd=0.1, subagent_fraction=0.5)
    with pytest.raises(TypeError):
        SpendBudgetMiddleware("5")  # type: ignore[arg-type]


def test_estimate_reads_a_real_chat_anthropic() -> None:
    from langchain.agents.middleware.types import ModelRequest

    anthropic = pytest.importorskip("langchain_anthropic")
    model = anthropic.ChatAnthropic(model="claude-sonnet-4-5", max_tokens=1024, api_key="unused")
    estimate = SpendBudgetMiddleware(1.0).estimate(
        ModelRequest(model=model, messages=[HumanMessage("hi")])
    )
    assert (estimate.provider, estimate.model, estimate.max_output_tokens) == (
        "anthropic",
        "claude-sonnet-4-5",
        1024,
    )
    assert estimate.cost_usd == pytest.approx(estimate.input_tokens * 6e-6 + 1024 * OUT_RATE)
