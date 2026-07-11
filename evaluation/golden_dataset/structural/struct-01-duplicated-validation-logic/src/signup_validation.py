"""Email validation used by the signup and login forms."""
import re

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email_signup(email: str) -> bool:
    if email is None:
        return False
    email = email.strip()
    if len(email) == 0:
        return False
    if len(email) > 254:
        return False
    if not _EMAIL_PATTERN.match(email):
        return False
    return True


def validate_email_login(email: str) -> bool:
    if email is None:
        return False
    email = email.strip()
    if len(email) == 0:
        return False
    if len(email) > 254:
        return False
    if not _EMAIL_PATTERN.match(email):
        return False
    return True
