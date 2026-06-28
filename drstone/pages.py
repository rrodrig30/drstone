"""Public pages + auth routes for Dr Stone: landing, login, register, logout.

Renders a professional light-theme UI (shared stylesheet at
/static/css/drstone.css) and wires server-side session cookies via drstone.auth.
"""

from __future__ import annotations

import html

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from drstone import auth

router = APIRouter()

_HEAD = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="icon" href="/static/img/logo.png">
<link rel="stylesheet" href="/static/css/drstone.css"></head><body>"""


def topbar(user) -> str:
    """Shared app header with logo + (login CTAs | user menu)."""
    if user:
        role = user.get("role", "user")
        badge = ('<span class="badge super">Superuser</span>' if role == "superuser"
                 else '<span class="badge">User</span>')
        name = html.escape(user.get("full_name") or user.get("email", ""))
        right = f"""<div class="usermenu">
          <div class="who"><b>{name}</b><span>{html.escape(user.get("email",""))}</span></div>
          {badge}
          <a class="btn-ghost" href="/logout">Sign out</a>
        </div>"""
    else:
        right = """<div class="usermenu">
          <a class="btn-ghost" href="/login">Sign in</a>
          <a class="btn" href="/register" style="padding:6px 14px;font-size:13px">Create account</a>
        </div>"""
    return f"""<div class="topbar"><div class="topbar-in">
      <a href="/"><img src="/static/img/logo.png" alt="Dr Stone"></a>
      <div class="brand">Dr&nbsp;Stone<small>Kidney-stone decision support</small></div>
      <div class="spacer"></div>{right}
    </div></div>"""


# --------------------------------------------------------------------------
# Landing
# --------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
def landing(request: Request):
    user = auth.current_user(request)
    cta = ('<a class="btn" href="/drstone" style="padding:13px 26px;font-size:16px">Open Dr&nbsp;Stone</a>'
           if user else
           '<a class="btn" href="/register" style="padding:13px 26px;font-size:16px">Create account</a>'
           '<a class="btn-ghost" href="/login" style="padding:13px 22px;font-size:15px;margin-left:10px">Sign in</a>')
    return HTMLResponse(_HEAD.format(title="Dr Stone — Kidney-Stone Decision Support") + topbar(user) + f"""
<div class="wrap" style="max-width:1040px">
  <div style="display:flex;gap:40px;align-items:center;flex-wrap:wrap;margin:18px 0 8px">
    <div style="flex:1;min-width:320px">
      <div style="color:var(--blue);font-weight:700;font-size:13px;letter-spacing:.08em;text-transform:uppercase">Point-of-care urolithiasis</div>
      <h1 style="font-size:34px;line-height:1.15;margin:8px 0 14px">Stone composition &amp; management,<br>from the scan you already have.</h1>
      <p style="font-size:16px;color:var(--ink-2);max-width:520px">From a non-contrast stone-protocol CT and routine ED labs, Dr&nbsp;Stone returns a
      <b>composition probability distribution</b> (calcium oxalate, calcium phosphate, uric acid, struvite, other),
      an <b>acute management</b> recommendation (medical expulsive therapy vs. intervention), and
      <b>tailored prevention</b> guidance — no dual-energy CT required.</p>
      <div style="margin-top:24px">{cta}</div>
    </div>
    <div style="flex:1;min-width:300px">
      <img src="/static/img/patient.jpg" alt="Patient presenting with renal colic in the emergency department"
           style="width:100%;border-radius:16px;box-shadow:var(--shadow);border:1px solid var(--line)">
    </div>
  </div>

  <div class="grid" style="margin-top:26px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))">
    <div class="card"><h3 style="color:var(--blue)">① Composition</h3>
      <div style="font-size:14px;color:var(--ink-2)">A ranked, calibrated probability across five stone types from CT stone density + metabolic labs.</div></div>
    <div class="card"><h3 style="color:var(--teal)">② Acute decision</h3>
      <div style="font-size:14px;color:var(--ink-2)">Size- and location-driven MET-vs-intervention guidance, with infection/obstruction red-flag alerts.</div></div>
    <div class="card"><h3 style="color:var(--green)">③ Prevention</h3>
      <div style="font-size:14px;color:var(--ink-2)">Per-type diet, medication, and lifestyle counseling plus a 24-hour-urine work-up prompt.</div></div>
  </div>

  <div class="disclaimer">Research / decision-support use; not a substitute for stone analysis, a 24-hour urine metabolic
  evaluation, or urology consultation. Acute and prevention guidance is draft content pending clinician sign-off.</div>
</div></body></html>""")


# --------------------------------------------------------------------------
# Auth shells
# --------------------------------------------------------------------------
def _auth_shell(title, heading, lead, body_form, alt) -> str:
    return _HEAD.format(title=title) + f"""
<div class="auth-shell">
  <div class="auth-hero">
    <img class="photo" src="/static/img/patient.jpg" alt="">
    <div class="hero-in">
      <div style="font-weight:700;letter-spacing:.1em;text-transform:uppercase;font-size:12px;color:#9cc7d6">Dr&nbsp;Stone</div>
      <h2>Decision support for stone disease, grounded in real-world ED data.</h2>
      <p>Composition probabilities, acute management, and prevention — from a non-contrast CT and routine labs.</p>
      <ul>
        <li>Calibrated composition distribution across five stone types</li>
        <li>MET-vs-intervention guidance with red-flag alerts</li>
        <li>Tailored diet, medication, and lifestyle counseling</li>
      </ul>
    </div>
  </div>
  <div class="auth-panel"><div class="auth-card">
    <img class="logo" src="/static/img/logo.png" alt="Dr Stone">
    <h1>{heading}</h1>
    <div class="lead">{lead}</div>
    {body_form}
    <div class="alt">{alt}</div>
  </div></div>
</div></body></html>"""


def _flash(msg, kind="err"):
    return f'<div class="flash {kind}">{html.escape(msg)}</div>' if msg else ""


def login_html(error="", email=""):
    form = f"""{_flash(error)}
    <form method="post" action="/login">
      <label>Email</label>
      <input name="email" type="email" autocomplete="username" required value="{html.escape(email)}" placeholder="you@institution.edu">
      <label>Password</label>
      <input name="password" type="password" autocomplete="current-password" required placeholder="••••••••">
      <button class="full" type="submit">Sign in</button>
    </form>"""
    return _auth_shell("Sign in — Dr Stone", "Welcome back",
                       "Sign in to access the clinical workspace.", form,
                       'No account? <a href="/register">Create one</a>')


def register_html(error="", email="", full_name=""):
    form = f"""{_flash(error)}
    <form method="post" action="/register">
      <label>Full name</label>
      <input name="full_name" type="text" autocomplete="name" value="{html.escape(full_name)}" placeholder="Jane Smith, MD">
      <label>Email</label>
      <input name="email" type="email" autocomplete="username" required value="{html.escape(email)}" placeholder="you@institution.edu">
      <label>Password</label>
      <input name="password" type="password" autocomplete="new-password" required placeholder="At least 8 characters">
      <label>Confirm password</label>
      <input name="confirm" type="password" autocomplete="new-password" required placeholder="Re-enter password">
      <button class="full" type="submit">Create account</button>
    </form>"""
    return _auth_shell("Create account — Dr Stone", "Create your account",
                       "Register to use Dr Stone's clinical workspace.", form,
                       'Already registered? <a href="/login">Sign in</a>')


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if auth.current_user(request):
        return RedirectResponse("/drstone", status_code=303)
    return HTMLResponse(login_html())


@router.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request):
    form = await request.form()
    email = str(form.get("email", "")).strip()
    password = str(form.get("password", ""))
    user = auth.authenticate(email, password)
    if not user:
        return HTMLResponse(login_html("Incorrect email or password.", email),
                            status_code=401)
    token = auth.create_session(user["id"])
    resp = RedirectResponse("/drstone", status_code=303)
    resp.set_cookie(auth.SESSION_COOKIE, token, max_age=auth.SESSION_MAX_AGE,
                    httponly=True, samesite="lax")
    return resp


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if auth.current_user(request):
        return RedirectResponse("/drstone", status_code=303)
    return HTMLResponse(register_html())


@router.post("/register", response_class=HTMLResponse)
async def register_submit(request: Request):
    form = await request.form()
    email = str(form.get("email", "")).strip()
    full_name = str(form.get("full_name", "")).strip()
    password = str(form.get("password", ""))
    confirm = str(form.get("confirm", ""))
    if password != confirm:
        return HTMLResponse(register_html("Passwords do not match.", email, full_name),
                            status_code=400)
    try:
        user = auth.create_user(email, password, full_name=full_name, role="user")
    except ValueError as e:
        return HTMLResponse(register_html(str(e), email, full_name), status_code=400)
    token = auth.create_session(user["id"])
    resp = RedirectResponse("/drstone", status_code=303)
    resp.set_cookie(auth.SESSION_COOKIE, token, max_age=auth.SESSION_MAX_AGE,
                    httponly=True, samesite="lax")
    return resp


@router.get("/logout")
def logout(request: Request):
    auth.delete_session(request.cookies.get(auth.SESSION_COOKIE))
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(auth.SESSION_COOKIE)
    return resp
