"""Google GenAI provider helpers.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Provider-wide and per-model helpers for the ``google_genai`` harness profile.
The provider profile applies to all ``google_genai:*`` models; the Gemini 3.1
Pro helpers layer on top to exercise every `_HarnessProfile` field as a sample
reference implementation.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

from deepagents.profiles._harness_profiles import _HarnessProfile, _register_harness_profile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.agents.middleware.types import AgentMiddleware

# ---------------------------------------------------------------------------
# Provider-wide helpers (google_genai)
# ---------------------------------------------------------------------------

GOOGLE_GENAI_MIN_VERSION = "4.2.1"
"""Minimum `langchain-google-genai` version required for the provider."""


def check_google_genai_version(spec: str) -> None:  # noqa: ARG001 — required by `pre_init` signature
    """Raise if `langchain-google-genai` is below the minimum version.

    Skipped when the package is not installed at all; `init_chat_model`
    will surface its own missing-dependency error downstream.

    Args:
        spec: Raw model spec string (unused; required by `pre_init` signature).

    Raises:
        ImportError: If the installed version is too old.
    """
    try:
        installed = pkg_version("langchain-google-genai")
    except PackageNotFoundError:
        return
    try:
        is_old = Version(installed) < Version(GOOGLE_GENAI_MIN_VERSION)
    except InvalidVersion:
        return
    if is_old:
        msg = (
            f"deepagents requires langchain-google-genai>={GOOGLE_GENAI_MIN_VERSION}, "
            f"but {installed} is installed. "
            f"Run: pip install 'langchain-google-genai>={GOOGLE_GENAI_MIN_VERSION}'"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Gemini 3.1 Pro per-model helpers (toy / sample implementation)
# ---------------------------------------------------------------------------


def _gemini31_pro_dynamic_kwargs() -> dict[str, Any]:
    """Build dynamic init kwargs for Gemini 3.1 Pro.

    Reads ``GOOGLE_GENAI_SAFETY_THRESHOLD`` from the environment at init time
    so operators can adjust safety-filter strictness without touching code.

    Returns:
        Dictionary of kwargs to merge on top of ``init_kwargs``.
    """
    kwargs: dict[str, Any] = {}
    threshold = os.environ.get("GOOGLE_GENAI_SAFETY_THRESHOLD")
    if threshold is not None:
        kwargs["safety_threshold"] = threshold
    return kwargs


def _gemini31_pro_extra_middleware() -> Sequence[AgentMiddleware]:
    """Build Gemini 3.1 Pro-specific middleware (deferred import).

    Returns a `PatchToolCallsMiddleware` as a stand-in — a real profile would
    swap this for provider-specific middleware (e.g. grounding validation,
    citation enforcement).

    Returns:
        Single-element sequence with a placeholder middleware instance.
    """
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware  # noqa: PLC0415

    return [PatchToolCallsMiddleware()]


_GEMINI31_PRO_BASE_SYSTEM_PROMPT = """\
You are a Deep Agent powered by Gemini 3.1 Pro.

Be concise. Act, don't narrate. Batch independent tool calls."""
"""Compact base prompt replacing `BASE_AGENT_PROMPT` for Gemini 3.1 Pro.

A real profile might trim the default prompt for models that follow
instructions well with less scaffolding.
"""

# ---------------------------------------------------------------------------
# Profile registration
# ---------------------------------------------------------------------------

# Provider-wide defaults for all google_genai:* models. Per-model profiles
# inherit these via the merge mechanism.
_register_harness_profile(
    "google_genai",
    _HarnessProfile(
        # convert_system_message_to_human: older Gemini models required this;
        # modern ones handle system messages natively.
        init_kwargs={"convert_system_message_to_human": False},
        # pre_init: version gate — runs before init_chat_model.
        pre_init=check_google_genai_version,
    ),
)

# Layered on top of the "google_genai" provider profile — inherits
# convert_system_message_to_human=False and the version gate via the merge
# mechanism.
_register_harness_profile(
    "google_genai:gemini-3.1-pro",
    _HarnessProfile(
        # init_kwargs_factory: deferred kwargs from env vars.
        init_kwargs_factory=_gemini31_pro_dynamic_kwargs,
        # base_system_prompt: replaces BASE_AGENT_PROMPT entirely.
        base_system_prompt=_GEMINI31_PRO_BASE_SYSTEM_PROMPT,
        # system_prompt_suffix: appended after base_system_prompt.
        system_prompt_suffix=(
            "You have access to parallel tool execution. "
            "When multiple tool calls are independent, batch them "
            "into a single response to minimize round-trips."
        ),
        # tool_description_overrides: per-tool rewrites.
        tool_description_overrides={
            "task": "Delegate a subtask to a specialized subagent. Prefer launching independent subtasks concurrently.",
        },
        # extra_middleware: appended to every middleware stack.
        extra_middleware=_gemini31_pro_extra_middleware,
    ),
)
