"""Release comparison: verify produced artifacts against the registered baseline."""

from dwts_reproduction.release.compare import (
    CheckResult,
    compare,
    parse_tolerance,
    summarize,
)

__all__ = ["CheckResult", "compare", "parse_tolerance", "summarize"]
