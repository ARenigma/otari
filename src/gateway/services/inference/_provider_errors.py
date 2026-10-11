"""How a provider failure answers a caller: its status, detail, error code and headers."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from http import HTTPStatus
from typing import NamedTuple

from any_llm.exceptions import AnyLLMError, ContextLengthExceededError, InvalidRequestError, UnsupportedParameterError
from any_llm.types.completion import CompletionParams
from any_llm.types.messages import MessagesParams
from any_llm.types.responses import ResponsesParams

from gateway.core.error_codes import (
    CONTEXT_LENGTH_EXCEEDED,
    PROVIDER_CREDENTIAL_REJECTED,
    PROVIDER_ERROR,
    UPSTREAM_RATE_LIMITED,
    error_code_of,
    error_headers,
)
from gateway.core.provider_params import SENSITIVE_PARAM_FIELDS
from gateway.services.inference._upstream_errors import (
    is_provider_billing_error,
    is_provider_credential_rejection,
    upstream_error_message,
    upstream_exception_chain,
    upstream_exception_shape,
    upstream_retry_after,
)
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.provider_kwargs import missing_credential
from gateway.services.upstream_redaction import redact_upstream_message
from gateway.streaming import IncompleteStreamError

# Client-visible API contract values; do not edit them without a deprecation plan.
PROVIDER_TIMEOUT_DETAIL = "LLM provider timeout"
# Fixed details for the provider failures that are the gateway's fault rather
# than the caller's. These never embed the upstream message: a rejected
# credential or an exhausted account is where provider internals concentrate,
# and it is not the caller's problem to debug (see classify_provider_error and
# test_error_detail_leakage). The bad-request, model-not-found, and rate-limited
# details below are fallbacks, used only when the provider gave us no message.
PROVIDER_BAD_REQUEST_DETAIL = "The provider rejected the request as invalid (check the model name and parameters)"
PROVIDER_MODEL_NOT_FOUND_DETAIL = "The requested model was not found on the provider"
PROVIDER_CREDENTIALS_DETAIL = "The provider rejected the gateway's credentials"
PROVIDER_BILLING_DETAIL = (
    "The upstream provider account is out of credit or over its billing limit. "
    "Top up the provider account, or route this model to a provider that has credit."
)
PROVIDER_ACCOUNT_QUOTA_DETAIL = (
    "The upstream provider account is over one of its service quotas. "
    "Raise the quota, or route this model to another provider."
)
PROVIDER_RATE_LIMITED_DETAIL = "The provider rate-limited this request"


class ProviderErrorMapping(NamedTuple):
    """A safe, client-facing (status, detail) for a classified provider failure."""

    status_code: int
    detail: str


def _is_unsupported_feature_error(exc: BaseException) -> bool:
    """True when any-llm refused a request feature its backend cannot express.

    Newer any-llm releases use ``UnsupportedParameterError`` for typed provider
    capability checks, while older feature checks still raise
    ``NotImplementedError``. Unwrap ``original_exception`` as well so either
    signal survives the unified-exception wrapper.
    """
    unsupported_types = (NotImplementedError, UnsupportedParameterError)
    return any(isinstance(candidate, unsupported_types) for candidate in upstream_exception_chain(exc))


def _unsupported_feature_detail(exc: BaseException) -> str:
    """Return one actionable reason for a locally rejected request feature."""
    for candidate in upstream_exception_chain(exc):
        if isinstance(candidate, UnsupportedParameterError):
            # AnyLLMError.__str__ prefixes the provider to ``message``. Feeding
            # both through upstream_error_message would repeat the same reason.
            return redacted_upstream_detail(candidate.message, PROVIDER_BAD_REQUEST_DETAIL)
    return _upstream_message_detail(exc, PROVIDER_BAD_REQUEST_DETAIL)


def redacted_upstream_detail(message: str, fallback: str) -> str:
    """Return redacted explanatory text, or a fixed detail when none remains."""
    redacted = redact_upstream_message(message)
    explanatory = redacted.replace("[redacted]", "")
    if not any(char.isalpha() for char in explanatory):
        return fallback
    return redacted


_UNEXPECTED_KWARG = re.compile(r"unexpected keyword argument '([^']+)'")

# The params a caller can actually put in a request body, across the three
# formats that share this classifier. Derived from any-llm's ``*Params`` for the
# same reason the request schemas are (see ``_schema_derive``): a param any-llm
# grows is picked up here without an edit, and nothing else can pass the gate.
#
# Derived the same way means derived *minus* ``SENSITIVE_PARAM_FIELDS``, not
# merely from the same models. ``derive_request_base`` skips those names because
# a future any-llm version could add a credential or provider-selection field to
# a typed ``*Params``, and exposing one as caller-settable would let a request
# override the operator's value (#160). A name the schema refuses to accept is by
# definition a name no caller sent, so it must not pass the caller-fault gate
# either: the two definitions of "settable by a caller" are one definition, and
# spelling it twice is how they drift.
_FORWARDED_PARAMS: frozenset[str] = frozenset(
    (set(CompletionParams.model_fields) | set(MessagesParams.model_fields) | set(ResponsesParams.model_fields))
    - SENSITIVE_PARAM_FIELDS
)


def _provider_rejected_param_detail(param: str) -> str:
    """Detail for a request param the provider serving this model cannot express."""
    return f"The provider serving this model does not accept the '{param}' parameter"


def _rejected_param(exc: BaseException, request_params: Collection[str] = _FORWARDED_PARAMS) -> str | None:
    """The request param a provider SDK refused as an unknown keyword, if any.

    any-llm hands a provider's SDK the params its ``*Params`` model declares, so
    a param that model carries but the SDK's method does not take arrives as
    ``TypeError: ...create() got an unexpected keyword argument 'x'``. That is
    every OpenAI-only chat param against a provider that never grew one
    (``seed``, ``n``, the penalties, the logprobs family, against Anthropic).

    It carries no HTTP status, so it would otherwise fall through to the generic
    502 and report a permanent, caller-fixable mismatch between a model and a
    param as an upstream outage. Only the param name is returned: the SDK's
    method name is an internal detail and never reaches the client (see
    ``test_error_detail_leakage``).

    Two gates keep that from blaming a caller for someone else's fault, because
    Python raises this same wording for any bad keyword and this classifier is
    reached from ``except Exception`` arms that wrap more than the provider call:

    * The name has to be one a request body can carry: ``request_params``, which
      defaults to the completion formats' :data:`_FORWARDED_PARAMS`. A
      pass-through endpoint names the fields its own request set instead.
      An operator's ``client_args`` typo (``timeoutt``) and a gateway-internal
      signature drift after an SDK bump (a tool backend's ``image``) are neither,
      so they keep the generic 502 and stay on the error-rate panel as the
      upstream-or-gateway failures they are.
    * The failing callable must not be a constructor. ``client_args`` is
      operator-owned and reaches a provider *client's* ``__init__``, so a key
      there that happens to collide with a real param name (``client_args={"seed":
      1}``) would otherwise pass the first gate and read as the caller's ``seed``.

    Both gates key on the same interpreter wording the match already depends on,
    so neither adds a new assumption about how CPython phrases the error.
    """
    for candidate in upstream_exception_chain(exc):
        if not isinstance(candidate, TypeError):
            continue
        message = str(candidate)
        if "__init__" in message:
            continue
        match = _UNEXPECTED_KWARG.search(message)
        if match and match.group(1) in request_params:
            return match.group(1)
    return None


def _upstream_message_detail(exc: BaseException, fallback: str) -> str:
    """The detail for a failure the caller can act on.

    Returns the provider's own message, redacted and length-capped, because the
    provider is the only party that knows what it objected to.

    A message the provider put on an exception wins over the joined chain.
    An SDK's string form can repeat that message, add its own labels, or append the response body.
    A wrapper's message is its original's string form, so a message deeper in the chain wins over it.

    Falls back to ``fallback`` when what is left says nothing. Some SDKs
    stringify a failure as bare punctuation or the status code itself, and
    "404" is not an explanation the caller did not already have from the
    status. Requiring one letter is a low bar deliberately: it rejects the
    empty cases without second-guessing a provider that wrote a real sentence.
    A message made entirely of redaction placeholders is empty for this
    purpose, too.
    """
    wrapper_message: str | None = None
    for candidate in upstream_exception_chain(exc):
        message = _carried_message(candidate)
        if message is None:
            continue
        if getattr(candidate, "original_exception", None) is not None:
            wrapper_message = wrapper_message or message
        elif detail := redacted_upstream_detail(message, ""):
            return detail
    if wrapper_message is not None and (detail := redacted_upstream_detail(wrapper_message, "")):
        return detail
    return redacted_upstream_detail(upstream_error_message(exc), fallback)


def _carried_message(exc: BaseException) -> str | None:
    """The message the provider put on one exception, without text the SDK adds, or ``None``."""
    message = getattr(exc, "message", None)
    if not isinstance(message, str):
        message = _error_response(exc).get("Message")
    return message if isinstance(message, str) else None


def _error_response(exc: BaseException) -> Mapping[str, object]:
    """The ``Error`` part of an AWS SDK error response, or an empty mapping."""
    response = getattr(exc, "response", None)
    error = response.get("Error") if isinstance(response, Mapping) else None
    return error if isinstance(error, Mapping) else {}


def _is_account_quota_error(exc: BaseException) -> bool:
    """Whether the provider refused because the operator's account is over a service quota."""
    return any(
        _error_response(current).get("Code") == "ServiceQuotaExceededException"
        for current in upstream_exception_chain(exc)
    )


def classify_provider_error(
    exc: BaseException, *, request_params: Collection[str] = _FORWARDED_PARAMS
) -> ProviderErrorMapping | None:
    """Map an upstream provider exception to a safe, specific (status, detail).

    Returns ``None`` when the failure carries no signal we can safely act on, so
    the caller falls back to its existing generic provider-error response. The
    mapping is intentionally conservative, classifying only the cases a caller
    can act on and leaving everything else (including provider 5xx and
    connection errors) to the generic 502.

    Detail text splits on whether the caller can act on the failure. A rejection
    of the caller's request (400/422/404) and a rate limit (429) pass the
    provider's own message through, redacted and length-capped, because it is
    the only description of what was actually wrong. When the failure is the
    gateway's own (a rejected credential, an exhausted provider account, a 5xx),
    the detail stays a fixed string: those carry no remedy the caller could
    apply, and are where a raw message is most likely to name the operator's
    credentials or topology.

    Timeout detection (including the OpenAI/Anthropic SDKs' own
    ``APITimeoutError``, and a duck-typed fallback for other provider SDKs) is
    shared with the hybrid-mode fallback classifier via
    :func:`upstream_exception_shape`, so both stay in sync.

    ``request_params`` is the set of names a caller could have sent, which is
    what lets an SDK's unknown-keyword ``TypeError`` read as the caller's 400
    (see :func:`_rejected_param`).
    """
    missing = missing_credential(exc)
    if missing is not None:
        return ProviderErrorMapping(missing.status_code, missing.detail)
    kind, status_code = upstream_exception_shape(exc)
    if kind == "timeout":
        return ProviderErrorMapping(HTTPStatus.GATEWAY_TIMEOUT, PROVIDER_TIMEOUT_DETAIL)
    # any-llm raises UnsupportedParameterError for typed provider-capability
    # checks and still uses NotImplementedError for older feature checks such as
    # context_management/betas (#530). Neither carries an HTTP status, so either
    # would otherwise fall through to a generic 502/500 and tell the caller a
    # guaranteed-permanent failure was transient. The exception type is the
    # whole signal here, and its message names the unsupported feature.
    if _is_unsupported_feature_error(exc):
        return ProviderErrorMapping(HTTPStatus.BAD_REQUEST, _unsupported_feature_detail(exc))
    # The same shape one layer down: a param any-llm forwards that the resolved
    # provider's SDK has no parameter for.
    if (param := _rejected_param(exc, request_params)) is not None:
        return ProviderErrorMapping(HTTPStatus.BAD_REQUEST, _provider_rejected_param_detail(param))
    if status_code is None:
        # A status-less InvalidRequestError means the provider (gemini, bedrock,
        # anthropic) rejected the request shape before assigning an HTTP status.
        # Mirror the UnsupportedParameterError / NotImplementedError branches: the
        # exception type is the whole signal, and the provider's own message names
        # what was wrong. Gated on the failure's own status rather than each
        # link's, so a chain that does carry one (401, 429, 5xx) keeps its fixed
        # detail instead of echoing the upstream text as a client 400.
        if any(isinstance(candidate, InvalidRequestError) for candidate in upstream_exception_chain(exc)):
            return ProviderErrorMapping(
                HTTPStatus.BAD_REQUEST, _upstream_message_detail(exc, PROVIDER_BAD_REQUEST_DETAIL)
            )
        return None
    # Account billing exhaustion, which several providers report as a 400/422
    # rather than the 402 the condition deserves (and DeepSeek does report as a
    # 402). This is a gateway-side account fault rather than anything wrong with
    # the caller's request, so it surfaces as a 502 and never as a client-facing
    # 400: telling the caller to "check the model name and parameters" for an
    # empty wallet sends operators debugging the wrong thing. Checked ahead of the
    # status branches so it wins over both the generic bad-request detail and the
    # 402 fall-through to a bare 502.
    if is_provider_billing_error(exc):
        return ProviderErrorMapping(HTTPStatus.BAD_GATEWAY, PROVIDER_BILLING_DETAIL)
    # An account quota is the operator's limit, like billing, even where the provider answers 400.
    if _is_account_quota_error(exc):
        return ProviderErrorMapping(HTTPStatus.BAD_GATEWAY, PROVIDER_ACCOUNT_QUOTA_DETAIL)
    if status_code in (400, 422):
        return ProviderErrorMapping(HTTPStatus.BAD_REQUEST, _upstream_message_detail(exc, PROVIDER_BAD_REQUEST_DETAIL))
    if status_code == 404:
        return ProviderErrorMapping(
            HTTPStatus.NOT_FOUND, _upstream_message_detail(exc, PROVIDER_MODEL_NOT_FOUND_DETAIL)
        )
    # A provider rejecting the gateway's credentials is a gateway-config fault,
    # not the caller's, so never a client-facing 401/403. Nor a 5xx: no retry
    # makes the key valid, so it is refused like a missing credential is.
    if is_provider_credential_rejection(exc):
        return ProviderErrorMapping(HTTPStatus.FAILED_DEPENDENCY, PROVIDER_CREDENTIALS_DETAIL)
    # A 429 is not the caller's request to fix but is theirs to act on, and only
    # the provider's text names the exhausted quota and the retry window. Still
    # drops the upstream Retry-After: the (status, detail) pair cannot carry it,
    # and on a streaming failure the headers are already flushed.
    if status_code == 429:
        return ProviderErrorMapping(
            HTTPStatus.TOO_MANY_REQUESTS, _upstream_message_detail(exc, PROVIDER_RATE_LIMITED_DETAIL)
        )
    return None


def provider_error_headers(exc: BaseException, status_code: int) -> dict[str, str] | None:
    """Response headers for a classified provider failure, or ``None``.

    On a 429, ``Otari-Error-Code: upstream_rate_limited`` (so a caller can tell
    the provider's limit from the gateway's own) and the upstream ``Retry-After``,
    the one piece of a provider's rate-limit response that its message body cannot
    always carry. Restricted to the 429: on the statuses that surface as a
    fixed-detail 502 the header would describe the gateway's own upstream
    account, which is not the caller's to read. A 424 names which of the
    gateway's credentials failed it, missing or rejected, and nothing more.
    """
    if status_code == HTTPStatus.BAD_REQUEST and _is_context_length_error(exc):
        return error_headers(CONTEXT_LENGTH_EXCEEDED)
    missing = missing_credential(exc)
    if missing is not None and status_code == missing.status_code:
        return error_headers(missing.code)
    if status_code == HTTPStatus.FAILED_DEPENDENCY and is_provider_credential_rejection(exc):
        return error_headers(PROVIDER_CREDENTIAL_REJECTED)
    if status_code != HTTPStatus.TOO_MANY_REQUESTS:
        return None
    headers = error_headers(UPSTREAM_RATE_LIMITED)
    retry_after = upstream_retry_after(exc)
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return headers


def _is_context_length_error(exc: BaseException) -> bool:
    """Whether any-llm classified the failure as a prompt too long for the model."""
    return any(isinstance(current, ContextLengthExceededError) for current in upstream_exception_chain(exc))


def refusal_code(exc: BaseException) -> str | None:
    """The ``Otari-Error-Code`` a provider failure ending a request stands for, or None.

    Read by the stream error events, which go out after the headers that would
    otherwise carry it. An error already rendered as an HTTP refusal carries its
    own code, so this is for every other failure.
    """
    mapping = classify_provider_error(exc)
    if mapping is None:
        return None
    return error_code_of(provider_error_headers(exc, mapping.status_code))


def stream_error_code(exc: BaseException) -> str | None:
    """The ``error.code`` of the event that ends a stream on ``exc``, or None.

    The refusal's own code where it has one, otherwise ``provider_error`` for any
    failure the provider caused, so a client can tell a provider failure from the
    gateway's own without reading the message. A failure
    :func:`classify_provider_error` recognizes counts as the provider's even
    when it carries no status, such as a parameter the provider's SDK refused.
    """
    code = refusal_code(exc)
    if code is not None:
        return code
    if _is_upstream_failure(exc) or classify_provider_error(exc) is not None:
        return PROVIDER_ERROR
    return None


def stream_error_message(exc: BaseException) -> str | None:
    """The message of the event that ends a stream on ``exc``, or None for the format's fixed one.

    The detail the same failure gets as an HTTP error, where
    :func:`classify_provider_error` gives it one, since that detail is already
    safe for the caller to read.
    """
    mapping = classify_provider_error(exc)
    return mapping.detail if mapping is not None else None


def _is_upstream_failure(exc: BaseException) -> bool:
    """Whether ``exc`` came from the provider rather than from the gateway."""
    if isinstance(exc, IncompleteStreamError) or upstream_exception_shape(exc) != (None, None):
        return True
    return any(isinstance(current, AnyLLMError) for current in upstream_exception_chain(exc))


def failure_status_code(exc: BaseException, *, request_params: Collection[str] = _FORWARDED_PARAMS) -> int:
    """The HTTP status to record on the usage log for an upstream failure.

    Prefers the status the provider actually returned, which is deliberately not
    always the status the caller saw: an upstream 401/403 surfaces to the caller
    as a 424 (a provider rejecting the gateway's credentials is a gateway-config
    fault, not the caller's key), but the log keeps the 401 so "how much of my
    error rate is my own misconfiguration" stays answerable. When the provider
    returned no status at all (timeout, unreachable), records the gateway's own
    classification instead, so an error row still carries a code to group on.

    The tool-loop cap is checked first because it is the gateway's own limit, not
    an upstream failure: it carries no HTTP status of its own, so it would
    otherwise fall through to the generic 502 and read in the taxonomy as a
    provider outage. It reaches here from the streaming path, where the cap is
    raised while the SSE body is already streaming (see ``run_tool_loop_stream``)
    and settles through ``on_error``; the non-streaming path records the same 422
    at its own ``except MaxToolIterationsExceeded``.
    """
    if isinstance(exc, MaxToolIterationsExceeded):
        return HTTPStatus.UNPROCESSABLE_CONTENT
    # Before the wrapper's own status: a wrapper can carry a 500 around a
    # credential any-llm never found, and the row records what the caller saw.
    missing = missing_credential(exc)
    if missing is not None:
        return missing.status_code
    _kind, status_code = upstream_exception_shape(exc)
    if status_code is not None:
        return status_code
    mapping = classify_provider_error(exc, request_params=request_params)
    return mapping.status_code if mapping is not None else HTTPStatus.BAD_GATEWAY
