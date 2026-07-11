from src.signup_validation import validate_email_login, validate_email_signup


def test_signup_accepts_a_well_formed_email():
    assert validate_email_signup("new.user@example.com") is True


def test_signup_rejects_a_blank_email():
    assert validate_email_signup("") is False


def test_login_accepts_a_well_formed_email():
    assert validate_email_login("existing.user@example.com") is True


def test_login_rejects_a_missing_at_sign():
    assert validate_email_login("not-an-email") is False
