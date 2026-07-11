import sqlite3

from src.user_repository import create_users_table, find_user_by_username


def test_find_user_by_username_returns_matching_row():
    conn = sqlite3.connect(":memory:")
    create_users_table(conn)
    conn.execute("INSERT INTO users (username, email) VALUES ('alice', 'alice@example.com')")
    conn.commit()

    row = find_user_by_username(conn, "alice")

    assert row is not None
    assert row[1] == "alice"


def test_find_user_by_username_returns_none_for_missing_user():
    conn = sqlite3.connect(":memory:")
    create_users_table(conn)

    row = find_user_by_username(conn, "ghost")

    assert row is None
