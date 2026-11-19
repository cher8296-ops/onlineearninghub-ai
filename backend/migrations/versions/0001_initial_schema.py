"""Initial normalized OnlineEarningHub AI database schema.

Revision ID: 0001_initial_schema
Revises:
"""
import os

from alembic import op
from app.accounts import Base

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # The initial release provisions a clean database from the authoritative
    # SQLAlchemy metadata. Future schema changes must use explicit revisions.
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade():
    # Dropping the initial schema destroys user, billing, history, and audit data.
    # Require an explicit operator opt-in before any destructive rollback.
    if os.getenv("ALLOW_DESTRUCTIVE_DB_DOWNGRADE") != "1":
        raise RuntimeError(
            "Destructive database downgrade blocked. Set ALLOW_DESTRUCTIVE_DB_DOWNGRADE=1 "
            "only after taking and verifying a backup."
        )
    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=True)
