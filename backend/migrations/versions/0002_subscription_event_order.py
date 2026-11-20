"""Add Stripe subscription event ordering timestamp.

Revision ID: 0002_subscription_event_order
Revises: 0001_initial_schema
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_subscription_event_order"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS last_event_created BIGINT")


def downgrade():
    op.execute("ALTER TABLE subscriptions DROP COLUMN IF EXISTS last_event_created")
