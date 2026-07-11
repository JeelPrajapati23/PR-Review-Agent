from src.notification_sender import send_notification


def test_sends_notification_for_a_valid_email():
    result = send_notification("alice@example.com", "Your order has shipped")

    assert result is True


def test_rejects_an_invalid_email():
    result = send_notification("not-an-email", "Your order has shipped")

    assert result is False
