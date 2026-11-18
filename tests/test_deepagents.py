"""Parallel deepagents subagents spending from one budget."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from agent_budget import Budget
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_agent_budget import SpendBudgetMiddleware

from .conftest import IN_RATE, OUT_RATE, ScriptedChat, reply

deepagents = pytest.importorskip("deepagents")

CALL_COST = 20 * IN_RATE + 500 * OUT_RATE  # booked per call: $0.00756
HOLD = 20 * 6.0 / 1_000_000 + 500 * OUT_RATE  # reserved per call: $0.00762


def fixed_count(messages: Any) -> int:
    return 20  # deterministic holds for the test


class Script:
    """Main agent: one message with two `task` calls, then a final answer. Subagents: answer."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sub_calls: list[str] = []
        self.main_calls = 0

    def __call__(self, messages: list[BaseMessage]) -> AIMessage:
        first = next((m for m in messages if isinstance(m, HumanMessage)), None)
        text = str(first.content) if first else ""
        if text.startswith("SUB-"):
            with self.lock:
                self.sub_calls.append(text)
            return reply(f"{text} finished")
        with self.lock:
            self.main_calls += 1
        if any(m.type == "tool" for m in messages):
            return reply("all done")
        calls = [
            {
                "name": "task",
                "args": {"description": f"SUB-{x}", "subagent_type": "researcher"},
                "id": f"t-{x}",
            }
            for x in ("A", "B")
        ]
        return reply("delegating", tool_calls=calls)


def build(budget: Budget, script: Script, **kwargs: Any) -> Any:
    model = ScriptedChat(respond=script, delay=0.05)
    researcher = {
        "name": "researcher",
        "description": "Researches one question.",
        "system_prompt": "Research the question.",
        "model": model,
        "middleware": [SpendBudgetMiddleware(token_counter=fixed_count)],
    }
    return deepagents.create_deep_agent(
        model=model,
        subagents=[researcher],
        middleware=[SpendBudgetMiddleware(budget, token_counter=fixed_count, **kwargs)],
    )


def test_subagent_spend_is_booked_on_the_parent_budget() -> None:
    budget = Budget(1.00, name="orchestrator")
    script = Script()
    build(budget, script).invoke({"messages": [HumanMessage("go")]})
    assert sorted(script.sub_calls) == ["SUB-A", "SUB-B"]
    assert script.main_calls == 2
    assert budget.spent_usd == pytest.approx(4 * CALL_COST)
    assert budget.reserved_usd == 0


def test_parallel_subagents_cannot_both_take_the_last_slot() -> None:
    """Room for the delegating call plus one subagent call: one subagent is refused."""
    budget = Budget(CALL_COST + HOLD + 0.0001, name="orchestrator")
    script = Script()
    build(budget, script).invoke({"messages": [HumanMessage("go")]})
    assert len(script.sub_calls) == 1
    assert budget.spent_usd == pytest.approx(2 * CALL_COST)
    assert budget.spent_usd <= budget.limit_usd


def test_each_subagent_gets_its_own_slice() -> None:
    budget = Budget(1.00, name="orchestrator")
    script = Script()
    build(budget, script, subagent_usd=0.25).invoke({"messages": [HumanMessage("go")]})
    assert budget.delegations == 2
    assert budget.child_cost_usd == pytest.approx(2 * CALL_COST)
    assert budget.own_cost_usd == pytest.approx(2 * CALL_COST)
    assert budget.reserved_usd == 0  # both slices closed and refunded


def test_slice_smaller_than_one_call_stops_the_subagent() -> None:
    budget = Budget(1.00, name="orchestrator")
    script = Script()
    build(budget, script, subagent_usd=0.005).invoke({"messages": [HumanMessage("go")]})
    assert script.sub_calls == []
    assert budget.child_cost_usd == 0
