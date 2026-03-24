"""
auth.py
Authentication for Procurement Advisor.
Password gate + signed invite links (no login form needed for recipients).
"""

import os
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request, Response
from fastapi.responses import RedirectResponse

SITE_PASSWORD = os.getenv('SITE_PASSWORD', '')
SECRET_KEY = os.getenv('SECRET_KEY', 'change-me')
COOKIE_NAME = 'buybuy_session'
SESSION_MAX_AGE = 86400 * 7  # 7 days

LINK_MAX_AGE_DAYS = int(os.getenv('LINK_MAX_AGE_DAYS', '30'))

serializer = URLSafeTimedSerializer(SECRET_KEY)

# Separate serializer salt for invite links so they can't be confused with session tokens
INVITE_SALT = 'invite-link'


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


# ─── Invite Links ─────────────────────────────────────────────

def generate_invite_token(label: str = '', days: int = None) -> str:
    """Generate a signed invite token. Label is for your reference only."""
    if days is None:
        days = LINK_MAX_AGE_DAYS
    payload = {'access': True, 'label': label, 'days': days}
    return serializer.dumps(payload, salt=INVITE_SALT)


def validate_invite_token(token: str) -> dict:
    """Validate an invite token. Returns payload dict or None."""
    try:
        data = serializer.loads(token, salt=INVITE_SALT)
        # Check custom expiry
        max_age = data.get('days', LINK_MAX_AGE_DAYS) * 86400
        # Re-validate with the correct max_age
        data = serializer.loads(token, salt=INVITE_SALT, max_age=max_age)
        return data
    except (BadSignature, SignatureExpired):
        return None
