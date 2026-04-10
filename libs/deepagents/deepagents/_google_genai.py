"""Google GenAI provider helpers.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Sample per-model harness profile for ``google_genai:gemini-3.1-pro`` that
exercises every `HarnessProfile` field. Intended as a reference implementation
— a real profile would replace the placeholder values with production logic.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.agents.middleware.types import AgentMiddleware


GEMINI31_PRO_MIN_LANGCHAIN_GOOGLE_GENAI = "4.2.1"
"""Minimum `langchain-google-genai` version required for Gemini 3.1 Pro."""


def check_langchain_google_genai_for_gemini31(spec: str) -> None:  # noqa: ARG001 — required by `pre_init` signature
    """Raise if `langchain-google-genai` is below the minimum for Gemini 3.1 Pro.

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
        is_old = Version(installed) < Version(GEMINI31_PRO_MIN_LANGCHAIN_GOOGLE_GENAI)
    except InvalidVersion:
        return
    if is_old:
        msg = (
            f"google_genai:gemini-3.1-pro requires "
            f"langchain-google-genai>={GEMINI31_PRO_MIN_LANGCHAIN_GOOGLE_GENAI}, "
            f"but {installed} is installed. "
            f"Run: pip install 'langchain-google-genai>={GEMINI31_PRO_MIN_LANGCHAIN_GOOGLE_GENAI}'"
        )
        raise ImportError(msg)


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
