import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings
import logging

logger = logging.getLogger("repolens.database")

db_url = settings.DATABASE_URL
connect_args = {}

if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=settings.DEBUG
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _run_schema_migrations(bind_engine):
    """
    Idempotently check and migrate existing database tables (e.g. SQLite / PostgreSQL)
    to add newly introduced columns without dropping or modifying existing data.
    Fails fast if a required migration fails and leaves the schema incompatible with the ORM.
    """
    from sqlalchemy import text, inspect
    try:
        inspector = inspect(bind_engine)
        tables = inspector.get_table_names()
    except Exception as inspect_err:
        logger.error("Failed to inspect database tables during migration check: %s", inspect_err, exc_info=True)
        raise RuntimeError(f"Database inspection failed during schema migration check: {inspect_err}") from inspect_err

    if "repositories" in tables:
        columns = {col["name"] for col in inspector.get_columns("repositories")}
        migration_errors = []
        with bind_engine.connect() as conn:
            if "embedding_model" not in columns:
                try:
                    logger.info("Migrating repositories table: adding embedding_model column (legacy default 'unknown')...")
                    conn.execute(text("ALTER TABLE repositories ADD COLUMN embedding_model VARCHAR DEFAULT 'unknown'"))
                    conn.commit()
                except Exception as exc:
                    migration_errors.append(f"embedding_model: {exc}")
                    logger.error("Failed to add embedding_model column: %s", exc)
            else:
                # Ensure legacy records that have NULL for embedding_model are set to 'unknown',
                # without overwriting existing non-null embedding_model values.
                try:
                    conn.execute(text("UPDATE repositories SET embedding_model = 'unknown' WHERE embedding_model IS NULL"))
                    conn.commit()
                except Exception as exc:
                    logger.warning("Could not backfill NULL embedding_model values: %s", exc)

            if "embedding_dimension" not in columns:
                try:
                    logger.info("Migrating repositories table: adding embedding_dimension column...")
                    conn.execute(text("ALTER TABLE repositories ADD COLUMN embedding_dimension INTEGER DEFAULT NULL"))
                    conn.commit()
                except Exception as exc:
                    migration_errors.append(f"embedding_dimension: {exc}")
                    logger.error("Failed to add embedding_dimension column: %s", exc)

        # Post-migration schema verification:
        # Re-inspect table columns to ensure required ORM columns are present
        updated_inspector = inspect(bind_engine)
        updated_columns = {col["name"] for col in updated_inspector.get_columns("repositories")}
        required_cols = {"embedding_model", "embedding_dimension"}
        missing_cols = required_cols - updated_columns
        if missing_cols:
            err_msg = (
                f"Database schema migration failed. Required columns {missing_cols} are missing "
                f"from 'repositories' table and incompatible with the ORM models. "
                f"Errors encountered: {migration_errors}"
            )
            logger.critical(err_msg)
            raise RuntimeError(err_msg)

def init_db():
    logger.info("Initializing database tables...")
    if db_url.startswith("sqlite:///"):
        db_file = db_url.replace("sqlite:///", "")
        if db_file and not db_file.startswith(":memory:"):
            dir_name = os.path.dirname(db_file)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _run_schema_migrations(engine)

