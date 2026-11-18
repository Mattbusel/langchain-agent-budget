"""Agent middleware that reserves each model call's worst-case cost before the call."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import llm_prices
from agent_budget import Budget, BudgetError, BudgetExhausted, Reservation, current_budget
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_llm_prices import UnpricedModelError, price_message, resolve_price
from langgraph.types import Command

TokenCounter = Callable[[Sequence[BaseMessage]], int]

# Settings that cap a call's output tokens, by the name each provider package uses.
_MAX_OUTPUT_KEYS = ("max_tokens", "max_output_tokens", "max_completion_tokens", "max_new_tokens")


@dataclass(frozen=True)
class CallEstimate:
    """The most one model call can cost, worked out before the call is made.

    `cost_usd` is `None` when llm-prices has no price for the model.
    `max_output_tokens` is `None` when nothing caps the reply (no `max_tokens` setting and no
    model profile); the estimate then covers the input only.
    """

    provider: str | None
    model: str | None
    input_tokens: int
    max_output_tokens: int | None
    cost_usd: float | None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _ls_params(model: Any, settings: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(model._get_ls_params(**settings))
    except Exception:  # a custom chat model with an unusual signature
        return {}


def _max_output_tokens(model: Any, settings: dict[str, Any], ls: dict[str, Any]) -> int | None:
    """The output cap the call will be sent with, if any."""
    for key in _MAX_OUTPUT_KEYS:
        if (n := _positive_int(settings.get(key))) is not None:
            return n
    if (n := _positive_int(ls.get("ls_max_tokens"))) is not None:
        return n
    for key in _MAX_OUTPUT_KEYS:
        if (n := _positive_int(getattr(model, key, None))) is not None:
            return n
    return None


def _profile_max_output(model: Any) -> int | None:
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict):
        return _positive_int(profile.get("max_output_tokens"))
    return None


def _worst_case_usd(price: llm_prices.Price, input_tokens: int, output_tokens: int) -> float | None:
    """Price the input at the dearest input-side rate the row lists.

    Prompt caching can bill input as a cache write, which costs more than plain input
    (1.25x or 2x on Anthropic), so the worst case takes the highest of those rates.
    """
    candidates = [
        llm_prices.Usage(input=input_tokens, output=output_tokens),
        llm_prices.Usage(cache_write_5m=input_tokens, output=output_tokens),
        llm_prices.Usage(cache_write_1h=input_tokens, output=output_tokens),
    ]
    costs = []
    for usage in candidates:
        try:
            breakdown = price.cost_usage(usage)
        except Exception:
            breakdown = None
        if breakdown is not None:
            costs.append(float(breakdown.total))
    return max(costs) if costs else None


def _last_ai_message(response: Any) -> AIMessage | None:
    if isinstance(response, ExtendedModelResponse):
        response = response.model_response
    if isinstance(response, AIMessage):
        return response
    if isinstance(response, ModelResponse):
        for message in reversed(response.result):
            if isinstance(message, AIMessage):
                return message
    return None


@contextmanager
def _active(budget: Budget) -> Iterator[Budget]:
    """Make `budget` the current one without closing it on exit.

    `with scope:` on a child scope closes it when the block ends; here the scope belongs to
    the caller and must stay open, so only the base `Budget` enter/exit (which just set and
    reset `current_budget()`) are used.
    """
    Budget.__enter__(budget)
    try:
        yield budget
    finally:
        Budget.__exit__(budget, None, None, None)


class SpendBudgetMiddleware(AgentMiddleware[Any, Any]):
    """Hold a model call's worst-case dollar cost before making it, and book the real cost after.

    Before each model call the middleware works out the most the call can cost: the input
    tokens (approximate, from `token_counter`) plus the call's `max_tokens`, priced with
    llm-prices at the dearest input-side rate. It reserves that amount from an agent-budget
    `Budget`. If it does not fit, the call is not made. When the reply comes back, the hold is
    swapped for the reply's real cost from `usage_metadata`, and the unused part goes back.

    The check and the reservation are one step under the budget's lock, so parallel subagents
    and parallel runs sharing a budget can never be admitted for more than it has left.

    Example:
        ```python
        from agent_budget import Budget
        from langchain.agents import create_agent
        from langchain_agent_budget import SpendBudgetMiddleware

        budget = Budget(2.00, name="support-bot")
        agent = create_agent(
            "anthropic:claude-sonnet-4-6",
            tools=[...],
            middleware=[SpendBudgetMiddleware(budget)],
        )
        ```
    """

    def __init__(
        self,
        budget: Budget | float | None = None,
        *,
        exit_behavior: Literal["end", "error"] = "end",
        max_output_tokens: int | None = None,
        token_counter: TokenCounter | None = None,
        subagent_tools: Sequence[str] = ("task",),
        subagent_usd: float | None = None,
        subagent_fraction: float | None = None,
        provider: str | None = None,
        unpriced: Literal["warn", "ignore", "error"] = "warn",
    ) -> None:
        """Set the budget.

        Args:
            budget: An agent-budget `Budget` (or child scope) shared by every run of the
                agent, or a dollar amount to create one. `None`: use the budget that is
                current when the call is made (`with budget: agent.invoke(...)`, or the scope
                a parent agent opened for this subagent).
            exit_behavior: `"end"` answers with an `AIMessage` saying the budget is spent and
                makes no call, so the run ends; `"error"` raises the agent-budget error
                (`BudgetExhausted`, `BudgetKilled` or `ScopeClosed`).
            max_output_tokens: Output tokens to assume when the chat model sets no
                `max_tokens`. When neither is set, the model profile's maximum output is used.
            token_counter: Counts the input tokens of a request. Defaults to
                `count_tokens_approximately` (no network). Pass an exact counter, such as
                `model.get_num_tokens_from_messages`, for tighter holds.
            subagent_tools: Tools that run a subagent (deepagents' `task` by default). While
                one runs, the budget stays current, so a subagent with its own
                `SpendBudgetMiddleware()` (no budget argument) spends from it.
            subagent_usd: Give each subagent run its own slice of this many dollars, reserved
                from the budget before it starts and refunded when it ends.
            subagent_fraction: Give each subagent run a slice of this fraction (0, 1] of what
                is left. Set at most one of `subagent_usd` and `subagent_fraction`.
            provider: Provider to price with when the chat model does not report one
                (`ls_provider`). Any name llm-prices accepts.
            unpriced: When llm-prices has no price for the model: `"warn"` once and let the
                call through uncounted, `"ignore"`, or `"error"` (raise `UnpricedModelError`).
        """
        super().__init__()
        if isinstance(budget, bool):
            msg = "budget must be a Budget, a dollar amount or None"
            raise TypeError(msg)
        if isinstance(budget, (int, float)):
            budget = Budget(float(budget), name="agent")
        if budget is not None and not isinstance(budget, Budget):
            msg = "budget must be a Budget, a dollar amount or None"
            raise TypeError(msg)
        if exit_behavior not in {"end", "error"}:
            msg = f"Invalid exit_behavior: {exit_behavior!r}. Must be 'end' or 'error'"
            raise ValueError(msg)
        if unpriced not in {"warn", "ignore", "error"}:
            msg = f"Invalid unpriced: {unpriced!r}. Must be 'warn', 'ignore' or 'error'"
            raise ValueError(msg)
        if subagent_usd is not None and subagent_fraction is not None:
            msg = "Set at most one of subagent_usd and subagent_fraction"
            raise ValueError(msg)
        if max_output_tokens is not None and _positive_int(max_output_tokens) is None:
            msg = "max_output_tokens must be a positive int"
            raise ValueError(msg)
        self.budget = budget
        self.exit_behavior = exit_behavior
        self.max_output_tokens = max_output_tokens
        self.token_counter = token_counter
        self.subagent_tools = frozenset(subagent_tools)
        self.subagent_usd = subagent_usd
        self.subagent_fraction = subagent_fraction
        self.provider = provider
        self.unpriced = unpriced
        self._warned: set[str] = set()
        self._warn_lock = threading.Lock()

    # ── estimate ─────────────────────────────────────────────────────────
    def estimate(self, request: ModelRequest[Any]) -> CallEstimate:
        """The most `request` can cost, before it is sent."""
        settings = dict(request.model_settings or {})
        ls = _ls_params(request.model, settings)
        provider = self.provider or ls.get("ls_provider")
        model = ls.get("ls_model_name")
        messages: list[BaseMessage] = list(request.messages)
        if request.system_message is not None:
            messages.insert(0, request.system_message)
        if self.token_counter is not None:
            input_tokens = int(self.token_counter(messages))
        else:
            input_tokens = count_tokens_approximately(messages, tools=list(request.tools) or None)
        max_out = _max_output_tokens(request.model, settings, ls)
        if max_out is None:
            max_out = self.max_output_tokens or _profile_max_output(request.model)
        price = resolve_price(provider, model)
        cost = None if price is None else _worst_case_usd(price, input_tokens, max_out or 0)
        return CallEstimate(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            max_output_tokens=max_out,
            cost_usd=cost,
        )

    # ── helpers ──────────────────────────────────────────────────────────
    def _warn_once(self, key: str, text: str) -> None:
        with self._warn_lock:
            if key in self._warned:
                return
            self._warned.add(key)
        warnings.warn(text, stacklevel=3)

    def _scope(self) -> Budget | None:
        return current_budget() or self.budget

    def _refusal(self, error: BudgetError, estimate: CallEstimate) -> AIMessage:
        if isinstance(error, BudgetExhausted):
            text = (
                f"Spend budget reached: the next model call can cost up to "
                f"${error.requested_usd:.4f} and budget '{error.scope}' has "
                f"${error.remaining_usd:.4f} left."
            )
            if estimate.max_output_tokens:
                text += (
                    f" The estimate assumes up to {estimate.max_output_tokens:,} output tokens;"
                    " a lower max_tokens on the model lowers it."
                )
        else:
            text = f"Spend budget stopped the run: {error}"
        return AIMessage(content=text, additional_kwargs={"lc_source": "spend_budget"})

    def _reserve(self, request: ModelRequest[Any]) -> tuple[Reservation | AIMessage | None, Any]:
        scope = self._scope()
        if scope is None:
            msg = (
                "SpendBudgetMiddleware has no budget: pass one, or run the agent inside "
                "`with budget:` or under a parent agent's SpendBudgetMiddleware"
            )
            raise ValueError(msg)
        estimate = self.estimate(request)
        if estimate.cost_usd is None:
            label = f"{estimate.provider or 'unknown provider'}/{estimate.model or 'unknown model'}"
            if self.unpriced == "error":
                msg = f"No llm-prices price for {label}; cannot enforce the spend budget"
                raise UnpricedModelError(msg)
            if self.unpriced == "warn":
                self._warn_once(
                    f"unpriced:{label}",
                    f"SpendBudgetMiddleware: no llm-prices price for {label}; "
                    "its calls are not counted against the budget",
                )
            return None, estimate
        if estimate.max_output_tokens is None:
            self._warn_once(
                f"unbounded:{estimate.model}",
                f"SpendBudgetMiddleware: nothing caps the output of {estimate.model}; "
                "set max_tokens on the model (or max_output_tokens here) so the hold "
                "covers the reply too",
            )
        try:
            return scope.reserve(estimate.cost_usd), estimate
        except BudgetError as error:
            if self.exit_behavior == "error":
                raise
            return self._refusal(error, estimate), estimate

    def _settle(self, hold: Reservation, response: Any, estimate: CallEstimate) -> None:
        message = _last_ai_message(response)
        provider = self.provider or estimate.provider
        priced = None if message is None else price_message(message, provider=provider)
        if priced is not None and priced.cost is None and estimate.model and message is not None:
            # The reply names a model llm-prices does not list; price the one that was asked for.
            priced = price_message(message, provider=provider, model=estimate.model)
        if priced is None or priced.cost is None:
            self._warn_once(
                f"nousage:{estimate.model}",
                f"SpendBudgetMiddleware: {estimate.model} returned no usage_metadata; "
                "booking the reserved worst case",
            )
            hold.settle(hold.estimate_usd)
            return
        hold.settle(priced.cost)

    def _child(self, request: ToolCallRequest) -> Budget | ToolMessage | None:
        """A slice of the budget for one subagent run, or `None` for no slice."""
        if self.subagent_usd is None and self.subagent_fraction is None:
            return None
        scope = self._scope()
        if scope is None:
            return None
        name = request.tool_call.get("name") or "subagent"
        try:
            return scope.child(
                f"{name}:{request.tool_call.get('id') or ''}",
                usd=self.subagent_usd,
                fraction=self.subagent_fraction,
            )
        except BudgetError as error:
            if self.exit_behavior == "error":
                raise
            return ToolMessage(
                content=f"Subagent not started: {error}",
                tool_call_id=request.tool_call.get("id") or "",
                name=name,
                status="error",
            )

    # ── hooks ────────────────────────────────────────────────────────────
    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Reserve the call's worst case, make the call, settle its real cost."""
        hold, estimate = self._reserve(request)
        if hold is None:
            return handler(request)
        if isinstance(hold, AIMessage):
            return hold
        with hold:  # released, not booked, if the call raises
            response = handler(request)
            self._settle(hold, response, estimate)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Async `wrap_model_call`."""
        hold, estimate = self._reserve(request)
        if hold is None:
            return await handler(request)
        if isinstance(hold, AIMessage):
            return hold
        with hold:
            response = await handler(request)
            self._settle(hold, response, estimate)
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run subagent tools inside the budget, or inside a slice of it."""
        if request.tool_call.get("name") not in self.subagent_tools:
            return handler(request)
        child = self._child(request)
        if isinstance(child, ToolMessage):
            return child
        if child is not None:
            with child:  # closes the slice and refunds what the subagent did not spend
                return handler(request)
        if current_budget() is None and self.budget is not None:
            with _active(self.budget):
                return handler(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async `wrap_tool_call`."""
        if request.tool_call.get("name") not in self.subagent_tools:
            return await handler(request)
        child = self._child(request)
        if isinstance(child, ToolMessage):
            return child
        if child is not None:
            with child:
                return await handler(request)
        if current_budget() is None and self.budget is not None:
            with _active(self.budget):
                return await handler(request)
        return await handler(request)
