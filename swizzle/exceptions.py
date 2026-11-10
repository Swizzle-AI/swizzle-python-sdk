"""
Custom exceptions for the Swizzle Python SDK.
"""

class SwizzleError(Exception):
    """Base exception for all Swizzle SDK errors."""
    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(SwizzleError):
    """Raised when an API key is invalid or revoked."""
    pass


class RateLimitError(SwizzleError):
    """Raised when the tenant's RPM or TPM bucket is exhausted."""
    pass


class WorkerUnavailableError(SwizzleError):
    """Raised when no silicon workers are healthy in the target cluster pool."""
    pass

