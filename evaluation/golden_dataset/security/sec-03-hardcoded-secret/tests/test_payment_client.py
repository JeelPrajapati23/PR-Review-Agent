from unittest.mock import MagicMock

from src.payment_client import PaymentClient


def test_charge_returns_parsed_response():
    fake_http_client = MagicMock()
    fake_http_client.post.return_value.json.return_value = {"status": "succeeded", "id": "ch_123"}
    fake_http_client.post.return_value.raise_for_status.return_value = None

    client = PaymentClient(client=fake_http_client)
    result = client.charge(card_number="4242424242424242", amount_cents=1500)

    assert result["status"] == "succeeded"
    fake_http_client.post.assert_called_once()
