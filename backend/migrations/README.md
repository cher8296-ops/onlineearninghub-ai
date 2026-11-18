# Database migrations

- The database is PostgreSQL; do not use SQLite or in-memory fallbacks in production.
- Apply schema changes through Alembic revisions, not ad-hoc production DDL.
- The first migration creates the initial schema from SQLAlchemy metadata.
- Future migrations must contain explicit, reviewed upgrade and downgrade operations.
- Run from the backend service root: `alembic upgrade head`.
- Check the current version with: `alembic current`.
- Generate a candidate revision locally with: `alembic revision --autogenerate -m "describe change"`, then review it before committing.
- Never commit DATABASE_URL, database passwords, payment credentials, or customer data.
- Before destructive migrations, take and verify a database backup.
