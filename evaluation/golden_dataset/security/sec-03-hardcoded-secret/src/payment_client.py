"""Thin wrapper around the (fictional) Acme Payments HTTP API."""
import logging

import httpx

logger = logging.getLogger(__name__)

# TODO: move to a config file before the next release.
API_KEY = "sk_live_51Hc8x9K2mQpZtwXyAbCdEfGhIjKlMnOpQrStUvWxYz00Ff3Gh7Ij9Kl2Mn"
API_BASE_URL = "https://api.acmepayments.example/v1"


class PaymentClient:
    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(base_url=API_BASE_URL)

    def charge(self, card_number: str, amount_cents: int) -> dict:
        payload = {"card_number": card_number, "amount_cents": amount_cents, "api_key": API_KEY}
        logger.info("Submitting charge request: %s", payload)

        response = self._client.post("/charges", json=payload)
        response.raise_for_status()
        return response.json()
