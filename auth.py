"""
Modulo de autenticacao
"""
import hashlib

USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "demo": hashlib.sha256("demo123".encode()).hexdigest(),
    "user": hashlib.sha256("user123".encode()).hexdigest(),
}


def authenticate_user(username: str, password: str) -> bool:
    if username not in USERS:
        return False
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return USERS[username] == password_hash


def check_authentication():
    import streamlit as st
    return st.session_state.get('authenticated', False)
