"""Initial normalized OnlineEarningHub AI database schema.

Revision ID: 0001_initial_schema
Revises:
"""
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
    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=True)
