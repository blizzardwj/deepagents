"""Google GenAI provider helpers.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Provider-wide and per-model helpers for the ``google_genai`` harness profile.
The provider profile applies to all ``google_genai:*`` models; the Gemini 3.1
Pro helpers layer on top to exercise every `HarnessProfile` field as a sample
reference implementation.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

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


def gemini31_pro_dynamic_kwargs() -> dict[str, Any]:
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


def gemini31_pro_extra_middleware() -> Sequence[AgentMiddleware]:
    """Build Gemini 3.1 Pro-specific middleware (deferred import).

    Returns a `PatchToolCallsMiddleware` as a stand-in — a real profile would
    swap this for provider-specific middleware (e.g. grounding validation,
    citation enforcement).

    Returns:
        Single-element sequence with a placeholder middleware instance.
    """
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware  # noqa: PLC0415

    return [PatchToolCallsMiddleware()]


GEMINI31_PRO_BASE_SYSTEM_PROMPT = """\
You are a Deep Agent powered by Gemini 3.1 Pro.

Be concise. Act, don't narrate. Batch independent tool calls."""
"""Compact base prompt replacing `BASE_AGENT_PROMPT` for Gemini 3.1 Pro.

A real profile might trim the default prompt for models that follow
instructions well with less scaffolding.
"""
