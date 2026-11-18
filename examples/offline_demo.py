"""Runs with no API key: a post-call cap overshoots, a reservation does not.

A scripted model stands in for claude-sonnet-4-5. Every call costs $0.00756 at list price
(20 input and 500 output tokens) and the agent wants five tool rounds. The cap is $0.02.
"""

from __future__ import annotations

from typing import Any

from agent_budget import Budget
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_llm_prices import CostCapMiddleware

from langchain_agent_budget import SpendBudgetMiddleware


class Scripted(BaseChatModel):
    model_name: str = "claude-sonnet-4-5"
    max_tokens: int = 500
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "anthropic"
        return params

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.calls += 1
        rounds = sum(1 for m in messages if m.type == "tool")
        calls = [{"name": "lookup", "args": {"q": "x"}, "id": f"c{rounds}"}] if rounds < 5 else []
        message = AIMessage(
            content="working" if calls else "answer",
            tool_calls=calls,
            usage_metadata={"input_tokens": 20, "output_tokens": 500, "total_tokens": 520},
            response_metadata={"model_name": "claude-sonnet-4-5", "model_provider": "anthropic"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.bind(**kwargs)


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return "result"


CAP = 0.02

model = Scripted()
create_agent(model, tools=[lookup], middleware=[CostCapMiddleware(max_usd_per_run=CAP)]).invoke(
    {"messages": [HumanMessage("go")]}
)
print(f"check after each call: {model.calls} calls, ${model.calls * 0.00756:.5f} of ${CAP}")

model = Scripted()
budget = Budget(CAP)
result = create_agent(model, tools=[lookup], middleware=[SpendBudgetMiddleware(budget)]).invoke(
    {"messages": [HumanMessage("go")]}
)
print(f"reserve before each call: {model.calls} calls, ${budget.spent_usd:.5f} of ${CAP}")
print(result["messages"][-1].content)
