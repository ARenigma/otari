"""Gateway-owned response bodies for the pass-through endpoints, built from the provider's result."""

from decimal import Decimal

from any_llm.types.rerank import RerankResponse as ProviderRerankResponse

from gateway.core.metered_pricing import quantize_cost
from gateway.schemas.inference import InlineCost, RerankMeta, RerankResponse, RerankResult, RerankUsage
from gateway.services.pricing_service import ResolvedPricing, metered_search_units


def inline_cost(cost: Decimal | None, pricing: ResolvedPricing | None) -> InlineCost:
    """The ``cost_usd`` and ``pricing_source`` a pass-through response carries.

    ``cost`` is the amount written to the request's usage row and ``pricing``
    the rate that priced it. Both fields are set only when both are known, the
    rule a completion's inline usage follows, so an unpriced request carries
    neither.
    """
    if cost is None or pricing is None:
        return InlineCost()
    return InlineCost(cost_usd=f"{quantize_cost(cost):.6f}", pricing_source=pricing.source)


def rerank_response(
    result: ProviderRerankResponse, *, pricing: ResolvedPricing | None, cost: Decimal | None
) -> RerankResponse:
    """The rerank response for ``result``, with what the request was metered at and cost.

    ``meta`` is the provider's billing report as it sent it. ``usage`` reports
    the provider's tokens, the search units billed when the model is priced per
    request, and the inline cost.
    """
    billed_units = result.meta.billed_units if result.meta else None
    cost_fields = inline_cost(cost, pricing)
    return RerankResponse(
        id=result.id,
        model=result.model,
        results=[RerankResult(index=r.index, relevance_score=r.relevance_score) for r in result.results],
        meta=RerankMeta(billed_units=result.meta.billed_units, tokens=result.meta.tokens) if result.meta else None,
        usage=RerankUsage(
            total_tokens=result.usage.total_tokens if result.usage else None,
            search_units=metered_search_units(pricing.pricing if pricing else None, billed_units),
            cost_usd=cost_fields.cost_usd,
            pricing_source=cost_fields.pricing_source,
        ),
    )
