import { type OAuthStartResult, startOAuthSignIn } from "@/shared/api/client"

/**
 * Where the CSRF `state` waits while the browser is away at the provider.
 *
 * `sessionStorage` and not `localStorage`: the value is scoped to the tab that
 * started the flow and dies with it, so a stale state cannot outlive the tab or
 * be read by a second one. The gateway checks the state again against its own
 * record and the HttpOnly flow cookie (`src/gateway/services/oauth_service.py`);
 * this copy is what binds a callback to the tab that started it.
 */
const STATE_KEY = "otari.oauth.state"

/**
 * Set beside the state when the person accepted the published terms before
 * leaving, which a sign-in that creates their account has to carry back.
 */
const TERMS_KEY = "otari.oauth.terms-accepted"

/** Remember the state, and whether the terms were accepted, for the round trip about to start. */
export function rememberOAuthState(
  state: string,
  { termsAccepted = false }: { termsAccepted?: boolean } = {},
): void {
  // A single-use CSRF nonce rather than a credential: it authorizes nothing on
  // its own, since the exchange also needs the provider's code and the flow cookie.
  // codeql[js/clear-text-storage-of-sensitive-data]
  window.sessionStorage.setItem(STATE_KEY, state)
  if (termsAccepted) {
    window.sessionStorage.setItem(TERMS_KEY, "true")
  } else {
    window.sessionStorage.removeItem(TERMS_KEY)
  }
}

/** Take what was remembered, clearing it so one value answers one callback. */
export function takeOAuthState(): { state: string; termsAccepted: boolean } {
  const state = window.sessionStorage.getItem(STATE_KEY) ?? ""
  const termsAccepted = window.sessionStorage.getItem(TERMS_KEY) === "true"
  window.sessionStorage.removeItem(STATE_KEY)
  window.sessionStorage.removeItem(TERMS_KEY)
  return { state, termsAccepted }
}

/**
 * Start an OAuth round trip: ask the gateway for the consent screen, remember
 * what the callback needs, and leave.
 *
 * Returns the gateway's refusal when it will not start one; otherwise the tab
 * is already navigating to the provider. `window.location.assign` rather than a
 * router navigation, because the destination is the provider's own origin.
 */
export async function leaveForOAuthProvider(
  provider: string,
  { termsAccepted = false }: { termsAccepted?: boolean } = {},
): Promise<OAuthStartResult> {
  const started = await startOAuthSignIn(provider)
  if (started.isOk) {
    rememberOAuthState(started.state, { termsAccepted })
    window.location.assign(started.authorizationUrl)
  }
  return started
}
