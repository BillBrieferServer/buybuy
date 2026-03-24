"""
main.py
Procurement Advisor — FastAPI application.
buybuy.quietimpact.ai
"""

import os
import logging
import markdown
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import check_auth, verify_password, create_session_cookie, logout, generate_invite_token, validate_invite_token
from . import procurement as proc
from . import advisor
from . import knowledge_base as kb
from .rate_limit import RateLimiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Procurement Advisor", docs_url=None, redoc_url=None)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize cache on startup
advisor.init_cache()

# Rate limiter: 10 AI requests per minute per IP
ai_limiter = RateLimiter(max_requests=10, window_seconds=60)

CONV_COOKIE = "buybuy_conv"


def get_client_ip(request: Request) -> str:
    """Get client IP, respecting X-Forwarded-For from nginx."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


def get_conversation_id(request: Request) -> str:
    """Get or create a conversation ID from cookie."""
    return request.cookies.get(CONV_COOKIE) or advisor.new_conversation_id()


def set_conversation_cookie(response, conversation_id: str):
    """Set the conversation ID cookie."""
    response.set_cookie(
        CONV_COOKIE, conversation_id,
        max_age=86400,  # 24 hours
        httponly=True,
        samesite="lax",
        secure=True,
    )


# ─── Health Check ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "procurement-advisor"}


# ─── Public Routes ────────────────────────────────────────────


def render_history(history):
    """Convert assistant markdown to HTML in conversation history."""
    rendered = []
    for msg in history:
        if msg['role'] == 'assistant':
            rendered.append({
                'role': msg['role'],
                'content': markdown.markdown(msg['content'], extensions=['tables', 'fenced_code'])
            })
        else:
            rendered.append(msg)
    return rendered

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if check_auth(request):
        return RedirectResponse("/home", status_code=302)
    return templates.TemplateResponse(request, "index.html")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if check_auth(request):
        return RedirectResponse("/home", status_code=302)
    return templates.TemplateResponse(request, "login.html", context={"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, password: str = Form(...)):
    if verify_password(password):
        response = RedirectResponse("/home", status_code=302)
        create_session_cookie(response)
        return response
    return templates.TemplateResponse(request, "login.html", context={
        "error": "Incorrect password. Please try again."
    })


@app.get("/logout")
async def logout_route(request: Request):
    response = RedirectResponse("/", status_code=302)
    logout(response)
    return response

@app.get("/enter/{token}")
async def enter_via_link(request: Request, token: str):
    """Authenticate via signed invite link — no password form needed."""
    data = validate_invite_token(token)
    if not data:
        return templates.TemplateResponse(request, "login.html", context={
            "error": "This link has expired or is invalid. Please request a new one."
        })
    response = RedirectResponse("/home", status_code=302)
    create_session_cookie(response)
    return response


@app.get("/invite", response_class=HTMLResponse)
async def invite_page(request: Request):
    """Generate invite links (requires auth)."""
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "invite.html", context={
        "link": None
    })


@app.post("/invite", response_class=HTMLResponse)
async def invite_generate(request: Request, label: str = Form(''), days: int = Form(30)):
    """Generate a new invite link."""
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)
    days = max(1, min(days, 365))
    token = generate_invite_token(label=label, days=days)
    base_url = str(request.base_url).rstrip('/')
    link = f"{base_url}/enter/{token}"
    return templates.TemplateResponse(request, "invite.html", context={
        "link": link,
        "label": label,
        "days": days
    })



# ─── Authenticated Routes ─────────────────────────────────────

@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "home.html")


@app.get("/advisor", response_class=HTMLResponse)
async def advisor_page(request: Request):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)

    conversation_id = get_conversation_id(request)
    history = render_history(advisor.get_conversation(conversation_id))

    response = templates.TemplateResponse(request, "advisor.html", context={
        "history": history,
        "conversation_id": conversation_id,
        "question": None
    })
    set_conversation_cookie(response, conversation_id)
    return response


@app.post("/advisor", response_class=HTMLResponse)
async def advisor_submit(request: Request, question: str = Form(...)):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)

    # Rate limit check
    client_ip = get_client_ip(request)
    if not ai_limiter.is_allowed(client_ip):
        remaining_wait = int(ai_limiter.seconds_until_next(client_ip)) + 1
        conversation_id = get_conversation_id(request)
        history = render_history(advisor.get_conversation(conversation_id))
        response = templates.TemplateResponse(request, "advisor.html", context={
            "history": history,
            "conversation_id": conversation_id,
            "question": question,
            "rate_limited": True,
            "wait_seconds": remaining_wait
        })
        set_conversation_cookie(response, conversation_id)
        return response

    conversation_id = get_conversation_id(request)
    result = await advisor.ask(question, conversation_id=conversation_id)

    # Reload full conversation (now includes the new exchange)
    history = render_history(advisor.get_conversation(conversation_id))

    response = templates.TemplateResponse(request, "advisor.html", context={
        "history": history,
        "conversation_id": conversation_id,
        "question": question,
        "response": None,
    })
    set_conversation_cookie(response, conversation_id)
    return response


@app.get("/advisor/new", response_class=HTMLResponse)
async def advisor_new_conversation(request: Request):
    """Start a fresh conversation."""
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)
    new_id = advisor.new_conversation_id()
    response = RedirectResponse("/advisor", status_code=302)
    set_conversation_cookie(response, new_id)
    return response


@app.get("/decision-tree", response_class=HTMLResponse)
async def decision_tree_page(request: Request):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "decision_tree.html", context={
        "purchase_types": proc.PURCHASE_TYPES,
        "entity_types": proc.ENTITY_TYPES,
        "result": None
    })


@app.post("/decision-tree", response_class=HTMLResponse)
async def decision_tree_submit(request: Request,
                                purchase_type: str = Form(...),
                                amount: str = Form(...),
                                entity_type: str = Form('all'),
                                is_cooperative: str = Form('no')):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)

    try:
        amount_int = int(amount.replace(',', '').replace('$', '').strip())
    except ValueError:
        return templates.TemplateResponse(request, "decision_tree.html", context={
            "purchase_types": proc.PURCHASE_TYPES,
            "entity_types": proc.ENTITY_TYPES,
            "result": None,
            "error": "Please enter a valid dollar amount."
        })

    checklist_result = proc.generate_checklist(
        purchase_type=purchase_type,
        amount=amount_int,
        entity_type=entity_type,
        is_cooperative=(is_cooperative == 'yes'),
    )

    return templates.TemplateResponse(request, "decision_tree.html", context={
        "purchase_types": proc.PURCHASE_TYPES,
        "entity_types": proc.ENTITY_TYPES,
        "result": checklist_result,
        "submitted": {
            'purchase_type': purchase_type,
            'amount': amount_int,
            'entity_type': entity_type,
            'is_cooperative': is_cooperative
        }
    })
