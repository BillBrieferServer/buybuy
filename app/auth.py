"""
auth.py
Simple password-gate authentication for MVP.
Single shared password from .env, session cookie via itsdangerous.
"""

import os
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request, Response
from fastapi.responses import RedirectResponse

SITE_PASSWORD = os.getenv('SITE_PASSWORD', '')
SECRET_KEY = os.getenv('SECRET_KEY', 'change-me')
COOKIE_NAME = 'buybuy_session'
SESSION_MAX_AGE = 86400 * 7  # 7 days

serializer = URLSafeTimedSerializer(SECRET_KEY)


def create_session_cookie(response: Response):
    """Set an authenticated session cookie."""
    token = serializer.dumps({'authenticated': True})
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite='lax',
        secure=True,
    )
    return response


def check_auth(request: Request) -> bool:
    """Check if request has a valid session cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return False
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        return data.get('authenticated', False)
    except (BadSignature, SignatureExpired):
        return False


def verify_password(password: str) -> bool:
    """Check if the provided password matches."""
    return password == SITE_PASSWORD


def logout(response: Response):
    """Clear the session cookie."""
    response.delete_cookie(COOKIE_NAME)
    return response
