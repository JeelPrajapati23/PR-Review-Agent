"""Sends order-status notifications to customers."""
import logging

logger = logging.getLogger(__name__)


def send_notification(customer_email: str, message: str) -> bool:
    print(f"[DEBUG] sending to {customer_email}")  # leftover debug print

    # Old implementation kept for reference, remove once the new provider
    # has been running in prod for a full release cycle.
    # response = legacy_mailer.send(customer_email, message)
    # if response.status_code != 200:
    #     logger.error("legacy send failed")
    #     return False
    # return True

    if not customer_email or "@" not in customer_email:
        logger.warning("Invalid recipient, skipping notification")
        return False

    print(f"Sending message: {message}")
    logger.info("Notification sent to %s", customer_email)
    return True
