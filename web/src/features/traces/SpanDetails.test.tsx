import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { SpanDetails } from "@/features/traces/SpanDetails"
import { traceSpan } from "@/tests/traceFixtures"

describe("SpanDetails", () => {
  it("shows an output-only token count and never turns a missing one into zero", () => {
    render(
      <SpanDetails
        span={traceSpan({ kind: "llm", input_tokens: null, output_tokens: 7 })}
      />,
    )

    expect(screen.getByText(/unknown in · 7 out/)).toBeInTheDocument()
  })

  it("leaves out the tokens of a span that reported none", () => {
    render(<SpanDetails span={traceSpan()} />)

    expect(screen.queryByText("Tokens")).not.toBeInTheDocument()
  })
})
