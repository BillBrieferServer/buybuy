"""
main.py
Procurement Advisor — FastAPI application.
buybuy.quietimpact.ai
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import check_auth, verify_password, create_session_cookie, logout
from . import procurement as proc
from . import advisor
from . import knowledge_base as kb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Procurement Advisor", docs_url=None, redoc_url=None)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize cache on startup
advisor.init_cache()


def require_auth(request: Request):
    """Dependency that redirects to login if not authenticated."""
    if not check_auth(request):
        return None
    return True


# ─── Health Check ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "procurement-advisor"}


# ─── Public Routes ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if check_auth(request):
        return RedirectResponse("/home", status_code=302)
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if check_auth(request):
        return RedirectResponse("/home", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, password: str = Form(...)):
    if verify_password(password):
        response = RedirectResponse("/home", status_code=302)
        create_session_cookie(response)
        return response
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Incorrect password. Please try again."
    })


@app.get("/logout")
async def logout_route(request: Request):
    response = RedirectResponse("/", status_code=302)
    logout(response)
    return response


# ─── Authenticated Routes ─────────────────────────────────────

@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/advisor", response_class=HTMLResponse)
async def advisor_page(request: Request):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("advisor.html", {
        "request": request,
        "response": None,
        "question": None,
    })


@app.post("/advisor", response_class=HTMLResponse)
async def advisor_submit(request: Request, question: str = Form(...)):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)

    result = advisor.ask(question)
    import markdown
    response_html = markdown.markdown(
        result['response'],
        extensions=['tables', 'fenced_code']
    )

    return templates.TemplateResponse("advisor.html", {
        "request": request,
        "question": question,
        "response": response_html,
        "cached": result['cached'],
        "tokens": result['tokens'],
    })


@app.get("/decision-tree", response_class=HTMLResponse)
async def decision_tree_page(request: Request):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("decision_tree.html", {
        "request": request,
        "purchase_types": proc.PURCHASE_TYPES,
        "entity_types": proc.ENTITY_TYPES,
        "result": None,
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
        return templates.TemplateResponse("decision_tree.html", {
            "request": request,
            "purchase_types": proc.PURCHASE_TYPES,
            "entity_types": proc.ENTITY_TYPES,
            "result": None,
            "error": "Please enter a valid dollar amount.",
        })

    checklist_result = proc.generate_checklist(
        purchase_type=purchase_type,
        amount=amount_int,
        entity_type=entity_type,
        is_cooperative=(is_cooperative == 'yes'),
    )

    return templates.TemplateResponse("decision_tree.html", {
        "request": request,
        "purchase_types": proc.PURCHASE_TYPES,
        "entity_types": proc.ENTITY_TYPES,
        "result": checklist_result,
        "submitted": {
            'purchase_type': purchase_type,
            'amount': amount_int,
            'entity_type': entity_type,
            'is_cooperative': is_cooperative,
        },
    })


@app.get("/checklist", response_class=HTMLResponse)
async def checklist_page(request: Request):
    if not check_auth(request):
        return RedirectResponse("/login", status_code=302)
    return RedirectResponse("/decision-tree", status_code=302)


# ─── API Endpoints ────────────────────────────────────────────

@app.post("/api/ask")
async def api_ask(request: Request):
    if not check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    question = body.get('question', '')
    if not question:
        return JSONResponse({"error": "No question provided"}, status_code=400)
    result = advisor.ask(question)
    return JSONResponse(result)


@app.post("/api/evaluate")
async def api_evaluate(request: Request):
    if not check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    result = proc.evaluate_procurement(
        purchase_type=body.get('purchase_type', 'goods_services'),
        amount=int(body.get('amount', 0)),
        entity_type=body.get('entity_type', 'all'),
        is_cooperative=body.get('is_cooperative', False),
    )
    return JSONResponse(result)
