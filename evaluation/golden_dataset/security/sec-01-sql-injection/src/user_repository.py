"""Lookup helpers for the login flow."""
import sqlite3


def create_users_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, email TEXT)"
    )
    conn.commit()


def find_user_by_username(conn: sqlite3.Connection, username: str):
    """Return the user row matching ``username``, or ``None``."""
    query = "SELECT id, username, email FROM users WHERE username = '" + username + "'"
    cursor = conn.execute(query)
    return cursor.fetchone()
