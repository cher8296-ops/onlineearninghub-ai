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

## Render deployment checklist

1. Provision PostgreSQL in the same Render region as the backend (Oregon for the current service).
2. Copy the database's **Internal Database URL** from Render and set it as `DATABASE_URL` in the backend service Environment. Do not use the external URL for a same-region Render service.
3. Set the backend service's pre-deploy command to `alembic upgrade head`. If pre-deploy commands are unavailable, use the start command `alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT` for this single-instance deployment; move migrations back to a dedicated pre-deploy step before scaling horizontally.
4. Deploy and verify the service startup log reports PostgreSQL connectivity and Alembic schema verification.
5. Verify the schema using a read-only query: `SELECT version_num FROM alembic_version;` and inspect table names with `SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;`.
6. Configure `ADMIN_EMAILS` and SMTP settings separately. Never paste secrets into GitHub, source files, chat, or logs.

## Data protection and recovery

- The current Render Free Postgres resource is temporary and expires after 30 days. It is suitable only for development/testing, not production customer data.
- Before accepting real users or payments, move to a paid PostgreSQL plan with an explicit backup/PITR retention policy and confirm a restore procedure.
- Export before destructive migrations or database replacement. Store encrypted backups outside the database provider account, restrict access, and periodically test restoration.
- Define and publish retention/deletion policies for prompts, generated outputs, account records, payment references, and audit logs.
- Keep payment-card data entirely with the payment provider; store only provider IDs, plan status, and the minimum reconciliation metadata required.
- Use the internal URL for Render-to-Render traffic. External clients must use TLS.
- Keep pool sizes conservative and review connection limits before increasing Render service instances.
