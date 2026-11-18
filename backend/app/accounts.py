"""Account, persistence, billing, feedback and admin API for OnlineEarningHub AI.

All routes fail closed when their required production configuration is absent.
No in-memory database or fake payment success is used as a production fallback.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import smtplib
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Generator, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger("onlineearninghub.accounts")
router = APIRouter(tags=["accounts", "billing", "feedback", "admin"])

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://flowchera21.bg5.net").rstrip("/")
ADMIN_EMAILS = {x.strip().lower() for x in os.getenv("ADMIN_EMAILS", "").split(",") if x.strip()}
FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_AI_LIMIT", "10"))
PREMIUM_DAILY_LIMIT = int(os.getenv("PREMIUM_DAILY_AI_LIMIT", "1000"))
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7
VERIFY_TTL_SECONDS = 60 * 60 * 24

engine = None
SessionLocal = None
if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    is_verified: Mapped[bool] = mapped_column(default=False)
    plan: Mapped[str] = mapped_column(String(20), default="free")
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Verification(Base):
    __tablename__ = "email_verifications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AIRequest(Base):
    __tablename__ = "ai_requests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    tool: Mapped[str] = mapped_column(String(30), index=True)
    prompt: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    rating: Mapped[int] = mapped_column(Integer)
    message: Mapped[str] = mapped_column(String(2000), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    event: Mapped[str] = mapped_column(String(50), index=True)
    path: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class RegisterInput(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=12, max_length=128)


class LoginInput(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class ChatInput(BaseModel):
    message: str = Field(min_length=1, max_length=10000)


class FeedbackInput(BaseModel):
    rating: int = Field(ge=1, le=5)
    message: str = Field(default="", max_length=2000)


class AnalyticsInput(BaseModel):
    event: str = Field(pattern=r"^(page_view|tool_open|signup|login)$")
    path: str = Field(default="", max_length=300)


def initialize_database() -> bool:
    if engine is None:
        logger.warning("DATABASE_URL is not configured; account and persistence routes remain disabled.")
        return False
    Base.metadata.create_all(bind=engine)
    logger.info("Account and product database schema is ready.")
    return True


def get_db() -> Generator[Session, None, None]:
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Account storage is not configured yet. Please try again later.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _password_hash(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return base64.urlsafe_b64encode(salt).decode() + "$" + base64.urlsafe_b64encode(derived).decode()


def _password_valid(password: str, encoded: str) -> bool:
    try:
        salt_text, digest_text = encoded.split("$", 1)
        salt = base64.urlsafe_b64decode(salt_text.encode())
        expected = base64.urlsafe_b64decode(digest_text.encode())
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _token(payload: dict, secret: str, ttl: int) -> str:
    if not secret:
        raise HTTPException(status_code=503, detail="Authentication is not configured. Please contact the administrator.")
    header = _b64(b'{"alg":"HS256","typ":"JWT"}')
    claims = dict(payload)
    claims["exp"] = int(time.time()) + ttl
    body = _b64(json.dumps(claims, separators=(",", ":")).encode())
    signing = f"{header}.{body}".encode()
    signature = _b64(hmac.new(secret.encode(), signing, hashlib.sha256).digest())
    return f"{header}.{body}.{signature}"


def _decode_token(token: str) -> dict:
    if not JWT_SECRET:
        raise HTTPException(status_code=503, detail="Authentication is not configured yet.")
    try:
        header, body, signature = token.split(".")
        signing = f"{header}.{body}".encode()
        expected = _b64(hmac.new(JWT_SECRET.encode(), signing, hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if int(claims.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        return claims
    except (ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=401, detail="Your session is invalid or expired. Please sign in again.")


def _current_user(authorization: Optional[str], db: Session) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    claims = _decode_token(authorization.split(" ", 1)[1].strip())
    user = db.get(User, int(claims.get("sub", 0)))
    if not user or not user.is_verified:
        raise HTTPException(status_code=401, detail="Your account is unavailable. Please sign in again.")
    return user


def _send_email(recipient: str, subject: str, body: str) -> None:
    host = os.getenv("SMTP_HOST", "").strip()
    username = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("EMAIL_FROM", username).strip()
    if not all((host, username, password, sender)):
        raise HTTPException(status_code=503, detail="Email delivery is not configured. Account registration is temporarily unavailable.")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    try:
        with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=15) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(message)
    except (OSError, ValueError, smtplib.SMTPException) as exc:
        logger.error("Email delivery failed. ErrorType=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="We could not send the email right now. Please try again later.")


def _daily_usage(db: Session, user: User) -> int:
    start = datetime.combine(_now().date(), datetime.min.time(), tzinfo=timezone.utc)
    return db.scalar(select(func.count(AIRequest.id)).where(AIRequest.user_id == user.id, AIRequest.created_at >= start)) or 0


def _admin(user: User) -> None:
    if user.email.lower() not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Administrator access is required.")


def _stripe_request(path: str, fields: dict) -> dict:
    secret = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="Online payments are not configured yet.")
    data = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(
        "https://api.stripe.com/v1/" + path,
        data=data,
        headers={"Authorization": "Bearer " + secret, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        logger.error("Payment provider rejected request. Status=%s", exc.code)
        raise HTTPException(status_code=502, detail="The payment provider could not create a checkout session. No payment was taken.") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.error("Payment provider unavailable. ErrorType=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="The payment provider is temporarily unavailable. Please try again.") from exc


@router.get("/plans")
def plans():
    return {
        "plans": [
            {"id": "free", "name": "Free", "price": 0, "currency": "USD", "features": ["10 saved AI requests per day", "Four core AI tools", "Personal dashboard and request history", "Community support"]},
            {"id": "premium", "name": "Premium", "price": None, "currency": "USD", "billing": "monthly", "features": ["Higher usage limits", "Priority access when available", "Full request history", "Premium workspace features"], "availability": "Requires payment provider configuration"}
        ]
    }


@router.post("/auth/register")
def register(data: RegisterInput, db: Session = Depends(get_db)):
    email = data.email.strip().lower()
    if not re_email(email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if len(data.password) < 12 or not any(c.isalpha() for c in data.password) or not any(c.isdigit() for c in data.password):
        raise HTTPException(status_code=422, detail="Use at least 12 characters, including letters and numbers.")
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    # Fail before creating the user if verification email cannot be delivered.
    if not all(os.getenv(k, "").strip() for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD")):
        raise HTTPException(status_code=503, detail="Email verification is not configured yet. Registration is not available.")
    user = User(email=email, password_hash=_password_hash(data.password))
    db.add(user)
    db.flush()
    raw_token = secrets.token_urlsafe(32)
    db.add(Verification(user_id=user.id, token_hash=hashlib.sha256(raw_token.encode()).hexdigest(), expires_at=_now() + timedelta(seconds=VERIFY_TTL_SECONDS)))
    link = f"{FRONTEND_URL}/?verify={urllib.parse.quote(raw_token)}"
    try:
        _send_email(email, "Verify your OnlineEarningHub AI account", f"Welcome to OnlineEarningHub AI. Verify your email using this link:\n\n{link}\n\nThis link expires in 24 hours. If you did not create this account, ignore this email.")
    except HTTPException:
        db.rollback()
        raise
    db.commit()
    return {"status": "verification_sent", "message": "Check your email to verify your account before signing in."}


def re_email(value: str) -> bool:
    import re
    return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value))


@router.post("/auth/verify")
def verify_email(token: str, db: Session = Depends(get_db)):
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    record = db.scalar(select(Verification).where(Verification.token_hash == token_hash))
    if not record or record.expires_at < _now():
        raise HTTPException(status_code=400, detail="This verification link is invalid or expired. Register again or request a new link.")
    user = db.get(User, record.user_id)
    if not user:
        raise HTTPException(status_code=400, detail="This verification link is no longer valid.")
    user.is_verified = True
    db.delete(record)
    db.commit()
    return {"status": "verified", "message": "Your email is verified. You can now sign in."}


@router.post("/auth/login")
def login(data: LoginInput, db: Session = Depends(get_db)):
    email = data.email.strip().lower()
    user = db.scalar(select(User).where(User.email == email))
    if not user or not _password_valid(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Please verify your email before signing in.")
    token = _token({"sub": str(user.id), "email": user.email}, JWT_SECRET, TOKEN_TTL_SECONDS)
    return {"access_token": token, "token_type": "bearer", "expires_in": TOKEN_TTL_SECONDS, "user": {"id": user.id, "email": user.email, "plan": user.plan}}


@router.get("/me")
def me(authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    user = _current_user(authorization, db)
    return {"id": user.id, "email": user.email, "plan": user.plan, "is_verified": user.is_verified, "created_at": user.created_at.isoformat()}


@router.get("/me/dashboard")
def dashboard(authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    user = _current_user(authorization, db)
    used = _daily_usage(db, user)
    limit = PREMIUM_DAILY_LIMIT if user.plan == "premium" else FREE_DAILY_LIMIT
    total = db.scalar(select(func.count(AIRequest.id)).where(AIRequest.user_id == user.id)) or 0
    return {"user": {"email": user.email, "plan": user.plan}, "usage": {"today": used, "daily_limit": limit, "remaining": max(0, limit - used), "total_requests": total}, "recent_activity": history_rows(db, user.id, 10)}


def begin_ai_request(authorization: Optional[str]) -> Optional[int]:
    if not authorization:
        return None
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Account storage is not configured yet.")
    db = SessionLocal()
    try:
        user = _current_user(authorization, db)
        limit = PREMIUM_DAILY_LIMIT if user.plan == "premium" else FREE_DAILY_LIMIT
        used = _daily_usage(db, user)
        if used >= limit:
            raise HTTPException(status_code=429, detail=f"Daily {user.plan} plan limit reached ({limit} AI requests).")
        return user.id
    finally:
        db.close()


def record_ai_request(user_id: Optional[int], tool: str, prompt: str, result: dict) -> None:
    if user_id is None or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        db.add(AIRequest(user_id=user_id, tool=tool, prompt=prompt, answer=result.get("answer", ""), model=result.get("model", "")))
        db.add(AnalyticsEvent(user_id=user_id, event="ai_request", path="/" + tool))
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Could not persist AI request. Tool=%s", tool)
        raise HTTPException(status_code=503, detail="Your AI response was created, but we could not save it to your account. Please retry after checking your connection.")
    finally:
        db.close()


def history_rows(db: Session, user_id: int, limit: int) -> list:
    rows = db.scalars(select(AIRequest).where(AIRequest.user_id == user_id).order_by(AIRequest.created_at.desc()).limit(limit)).all()
    return [{"id": row.id, "tool": row.tool, "prompt": row.prompt, "answer": row.answer, "model": row.model, "created_at": row.created_at.isoformat()} for row in rows]


@router.get("/me/history")
def history(authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    user = _current_user(authorization, db)
    return {"items": history_rows(db, user.id, 100)}


@router.post("/chat")
def chat(data: ChatInput, authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    user = _current_user(authorization, db)
    limit = PREMIUM_DAILY_LIMIT if user.plan == "premium" else FREE_DAILY_LIMIT
    used = _daily_usage(db, user)
    if used >= limit:
        raise HTTPException(status_code=429, detail=f"Daily {user.plan} plan limit reached ({limit} AI requests).")
    # Import lazily to avoid a circular import; app.main owns the provider implementation.
    from . import main as ai
    result = ai.ask_ai(ai.GLOBAL_QUALITY_STANDARD + "\n\nYou are the OnlineEarningHub AI chat assistant. Answer clearly, safely and in the user's language.\n\n" + data.message)
    row = AIRequest(user_id=user.id, tool="chat", prompt=data.message, answer=result["answer"], model=result.get("model", ""))
    db.add(row)
    db.add(AnalyticsEvent(user_id=user.id, event="ai_request", path="/chat"))
    db.commit()
    return {"answer": result["answer"], "model_used": result.get("model", ""), "usage": {"today": used + 1, "daily_limit": limit, "remaining": max(0, limit - used - 1)}}


@router.post("/feedback")
def feedback(data: FeedbackInput, authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    user = _current_user(authorization, db) if authorization else None
    db.add(Feedback(user_id=user.id if user else None, rating=data.rating, message=data.message.strip()))
    db.commit()
    return {"status": "received", "message": "Thank you for helping us improve."}


@router.post("/analytics/event")
def analytics_event(data: AnalyticsInput, request: Request, authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    user = _current_user(authorization, db) if authorization else None
    # Store only allow-listed event names and normalized local paths; never store query strings or IP addresses.
    path = "/" + data.path.strip().lstrip("/")[:299] if data.path else ""
    db.add(AnalyticsEvent(user_id=user.id if user else None, event=data.event, path=path))
    db.commit()
    return {"status": "recorded"}


@router.post("/billing/checkout")
def billing_checkout(authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    user = _current_user(authorization, db)
    price_id = os.getenv("STRIPE_PRICE_PREMIUM_MONTHLY", "").strip()
    if not price_id:
        raise HTTPException(status_code=503, detail="Premium checkout is not configured yet.")
    session = _stripe_request("checkout/sessions", {
        "mode": "subscription",
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": "1",
        "customer_email": user.email,
        "client_reference_id": str(user.id),
        "metadata[user_id]": str(user.id),
        "success_url": FRONTEND_URL + "/?payment=success",
        "cancel_url": FRONTEND_URL + "/?payment=cancelled",
    })
    if not session.get("url"):
        raise HTTPException(status_code=502, detail="The payment provider did not return a checkout link.")
    return {"checkout_url": session["url"]}


@router.post("/billing/webhook")
async def billing_webhook(request: Request, db: Session = Depends(get_db)):
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="Payment webhook is not configured.")
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        parts = dict(item.split("=", 1) for item in signature.split(",") if "=" in item)
        timestamp = int(parts["t"])
        supplied = parts["v1"]
    except (KeyError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid payment signature.")
    if abs(int(time.time()) - timestamp) > 300:
        raise HTTPException(status_code=400, detail="Payment signature has expired.")
    expected = hmac.new(secret.encode(), str(timestamp).encode() + b"." + payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=400, detail="Invalid payment signature.")
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid payment event.")
    event_type = event.get("type")
    obj = (event.get("data") or {}).get("object") or {}
    if event_type == "checkout.session.completed" and obj.get("mode") == "subscription" and obj.get("payment_status") == "paid":
        user_id = (obj.get("metadata") or {}).get("user_id") or obj.get("client_reference_id")
        user = db.get(User, int(user_id)) if user_id and str(user_id).isdigit() else None
        if user:
            user.plan = "premium"
            user.stripe_customer_id = obj.get("customer")
            db.add(AnalyticsEvent(user_id=user.id, event="subscription_started", path="/billing"))
            db.commit()
    elif event_type == "customer.subscription.deleted":
        customer_id = obj.get("customer")
        user = db.scalar(select(User).where(User.stripe_customer_id == customer_id)) if customer_id else None
        if user:
            user.plan = "free"
            db.add(AnalyticsEvent(user_id=user.id, event="subscription_cancelled", path="/billing"))
            db.commit()
    return {"received": True}


@router.get("/admin/analytics")
def admin_analytics(authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    user = _current_user(authorization, db)
    _admin(user)
    since = _now() - timedelta(days=30)
    users = db.scalar(select(func.count(User.id))) or 0
    verified = db.scalar(select(func.count(User.id)).where(User.is_verified.is_(True))) or 0
    requests = db.scalar(select(func.count(AIRequest.id)).where(AIRequest.created_at >= since)) or 0
    feedback_count = db.scalar(select(func.count(Feedback.id)).where(Feedback.created_at >= since)) or 0
    page_views = db.scalar(select(func.count(AnalyticsEvent.id)).where(AnalyticsEvent.event == "page_view", AnalyticsEvent.created_at >= since)) or 0
    errors = 0  # Application errors are not stored as user analytics; use Render logs for operational errors.
    return {"period_days": 30, "users": users, "verified_users": verified, "ai_requests": requests, "feedback_count": feedback_count, "page_views": page_views, "application_errors": errors, "note": "Application errors are monitored in hosting logs; no prompt text or IP addresses are collected by analytics."}


@router.get("/admin/users")
def admin_users(authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    user = _current_user(authorization, db)
    _admin(user)
    rows = db.scalars(select(User).order_by(User.created_at.desc()).limit(200)).all()
    return {"users": [{"id": row.id, "email": row.email, "plan": row.plan, "verified": row.is_verified, "created_at": row.created_at.isoformat()} for row in rows]}
