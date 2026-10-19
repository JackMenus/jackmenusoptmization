"""Exception hierarchy used across the platform."""


class OptiforgeError(Exception):
    """Base class for all platform errors."""


class ProblemDefinitionError(OptiforgeError, ValueError):
    """The problem, space, or constraint definition is malformed."""


class UnsupportedProblemError(OptiforgeError):
    """A component was asked to handle a problem configuration it does not support."""


class BudgetExhausted(OptiforgeError):
    """An evaluation, time, or development budget was used up."""

    def __init__(self, kind: str, message: str = ""):
        super().__init__(message or f"budget exhausted: {kind}")
        self.kind = kind


class SearchCancelled(OptiforgeError):
    """The search was cancelled through a cancellation token."""


class CandidateRejected(OptiforgeError):
    """A generated candidate failed static checks or validation."""


class IsolationUnavailable(OptiforgeError):
    """Operating-system isolation for untrusted code is not available on this host."""
