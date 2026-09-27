import os
import re
from contextlib import contextmanager
from dataclasses import dataclass

import pymysql
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

_IDENT = re.compile(r"^[A-Za-z0-9_]{1,64}$")


class DatabaseConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class DbSettings:
    host: str
    port: int
    user: str
    password: str
    name: str
    charset: str
    collation: str
    pool_size: int
    max_overflow: int
    pool_recycle_seconds: int
    connect_timeout_seconds: int


def _load_db_settings() -> DbSettings:
    errors = []

    def req_str(name: str) -> str:
        v = (os.getenv(name) or "").strip()
        if not v:
            errors.append(f"{name} is required")
        return v

    def req_int(name: str, minimum: int = 1) -> int:
        v = req_str(name)
        if not v:
            return 0
        try:
            n = int(v)
        except ValueError:
            errors.append(f"{name} must be a whole number (got {v!r})")
            return 0
        if n < minimum:
            errors.append(f"{name} must be >= {minimum} (got {n})")
        return n

    def req_ident(name: str) -> str:
        v = req_str(name)
        if v and not _IDENT.match(v):
            errors.append(f"{name} may only contain letters, digits and underscore (got {v!r})")
        return v

    password = os.getenv("DB_PASSWORD")
    if password is None:
        errors.append("DB_PASSWORD must be present (leave it empty if the MySQL user has no password)")
        password = ""

    cfg = DbSettings(
        host=req_str("DB_HOST"),
        port=req_int("DB_PORT"),
        user=req_str("DB_USER"),
        password=password,
        name=req_ident("DB_NAME"),
        charset=req_ident("DB_CHARSET"),
        collation=req_ident("DB_COLLATION"),
        pool_size=req_int("DB_POOL_SIZE"),
        max_overflow=req_int("DB_MAX_OVERFLOW", minimum=0),
        pool_recycle_seconds=req_int("DB_POOL_RECYCLE_SECONDS"),
        connect_timeout_seconds=req_int("DB_CONNECT_TIMEOUT_SECONDS"),
    )

    if errors:
        raise DatabaseConfigError(
            "Invalid database configuration in .env:\n  - " + "\n  - ".join(errors) +
            "\nCopy .env.example to .env and fill in the values."
        )
    return cfg


def _ensure_database_exists(cfg: DbSettings) -> None:
    print(f"Connecting to MySQL at {cfg.host}:{cfg.port} as '{cfg.user}' "
          f"to ensure database '{cfg.name}' exists...")
    conn = None
    try:
        conn = pymysql.connect(
            host=cfg.host,
            port=cfg.port,
            user=cfg.user,
            password=cfg.password,
            charset=cfg.charset,
            connect_timeout=cfg.connect_timeout_seconds,
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{cfg.name}` "
                f"CHARACTER SET {cfg.charset} COLLATE {cfg.collation}"
            )
        conn.commit()
        print(f"MySQL database '{cfg.name}' ready (created if it didn't exist)")
    except pymysql.MySQLError as exc:
        raise DatabaseConfigError(
            f"Could not connect to MySQL / create database '{cfg.name}': {exc}\n"
            "Check DB_HOST, DB_PORT, DB_USER, DB_PASSWORD in .env, that the MySQL server is "
            "running, and that DB_USER is allowed to CREATE DATABASE."
        ) from exc
    finally:
        if conn is not None:
            conn.close()


db_settings = _load_db_settings()
_ensure_database_exists(db_settings)

engine = create_engine(
    URL.create(
        "mysql+pymysql",
        username=db_settings.user,
        password=db_settings.password,
        host=db_settings.host,
        port=db_settings.port,
        database=db_settings.name,
        query={"charset": db_settings.charset},
    ),
    pool_size=db_settings.pool_size,
    max_overflow=db_settings.max_overflow,
    pool_recycle=db_settings.pool_recycle_seconds,
    pool_pre_ping=True,
    connect_args={"connect_timeout": db_settings.connect_timeout_seconds},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _normalize_type_str(type_str: str, mariadb: bool = False) -> str:
    """
    Normalize a compiled/reflected column type string so equivalent types
    don't get falsely flagged as "out of sync" on every startup:

    - MySQL's information_schema reflects VARCHAR/TEXT columns with an
      explicit "COLLATE ..." (and sometimes "CHARACTER SET ...") suffix,
      but SQLAlchemy's own compiled model type never includes it — so a
      column that is actually fine would otherwise mismatch forever.
    - MySQL has no real BOOLEAN storage type; BOOL/BOOLEAN is just an
      alias for TINYINT(1). After ALTERing a column to BOOL, MySQL will
      always reflect it back as TINYINT(1) on the next inspection, so
      these two need to be treated as equivalent too.
    - Integer display widths (INTEGER(11)) are cosmetic and are ignored.
    - MariaDB stores JSON as LONGTEXT, so on MariaDB the two are equivalent.
    """
    s = re.sub(r"\s+COLLATE\s+\S+", "", type_str, flags=re.IGNORECASE)
    s = re.sub(r"\s+CHARACTER SET\s+\S+", "", s, flags=re.IGNORECASE)
    s = s.strip().upper()
    if s in ("BOOL", "BOOLEAN", "TINYINT(1)"):
        return "TINYINT(1)"
    s = re.sub(r"^(TINYINT|SMALLINT|MEDIUMINT|INTEGER|INT|BIGINT)\(\d+\)", r"\1", s)
    if mariadb and s == "JSON":
        s = "LONGTEXT"
    return s


def _auto_sync_mysql_columns() -> None:
    inspector = inspect(engine)
    mariadb = bool(getattr(engine.dialect, "is_mariadb", False))
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # brand new table — create_all() already built it correctly

            db_columns = {c["name"]: c for c in inspector.get_columns(table.name)}

            for column in table.columns:
                if column.primary_key:
                    continue  # never touch PK columns

                db_col = db_columns.get(column.name)
                if db_col is None:
                    # Column exists on the model but not yet in the live MySQL
                    # table (e.g. someone just added a new field to models.py).
                    # Add it automatically instead of requiring a manual ALTER.
                    try:
                        col_type_str = str(column.type.compile(dialect=engine.dialect))
                    except Exception as exc:
                        print(f"Could not compile type for new column "
                              f"'{table.name}.{column.name}': {exc}")
                        continue

                    nullability = "NULL" if column.nullable else "NOT NULL"
                    default_clause = ""
                    if column.default is not None and getattr(column.default, "is_scalar", False):
                        default_clause = f" DEFAULT {column.default.arg!r}"

                    print(f"Column '{table.name}.{column.name}' missing in MySQL. "
                          f"Auto-adding as {col_type_str}...")
                    try:
                        conn.execute(text(
                            f"ALTER TABLE `{table.name}` "
                            f"ADD COLUMN `{column.name}` {col_type_str} "
                            f"{nullability}{default_clause}"
                        ))
                        print(f"Column '{table.name}.{column.name}' added.")

                        if column.index or column.unique:
                            idx_name = f"ix_{table.name}_{column.name}"
                            unique_kw = "UNIQUE " if column.unique else ""
                            try:
                                conn.execute(text(
                                    f"CREATE {unique_kw}INDEX `{idx_name}` "
                                    f"ON `{table.name}` (`{column.name}`)"
                                ))
                                print(f"Index '{idx_name}' created.")
                            except Exception as exc:
                                print(f"Could not create index on "
                                      f"'{table.name}.{column.name}': {exc}")
                    except Exception as exc:
                        print(f"Could not auto-add column "
                              f"'{table.name}.{column.name}': {exc}")
                        print("You may need to add it manually with an ALTER TABLE statement.")
                    continue

                try:
                    model_type_str = str(column.type.compile(dialect=engine.dialect))
                    db_type_str = str(db_col["type"].compile(dialect=engine.dialect))
                except Exception:
                    continue  # if either type can't compile for this dialect, skip safely

                if _normalize_type_str(model_type_str, mariadb) == _normalize_type_str(db_type_str, mariadb):
                    continue  # already in sync (ignoring collation/charset noise + BOOL~TINYINT(1))

                nullability = "NULL" if column.nullable else "NOT NULL"
                print(f"Column '{table.name}.{column.name}' out of sync: "
                      f"{db_type_str} -> {model_type_str}. Auto-altering...")
                try:
                    conn.execute(text(
                        f"ALTER TABLE `{table.name}` "
                        f"MODIFY COLUMN `{column.name}` {model_type_str} {nullability}"
                    ))
                    print(f"Column '{table.name}.{column.name}' updated to {model_type_str}")
                except Exception as exc:
                    print(f"Could not auto-alter '{table.name}.{column.name}': {exc}")
                    print("You may need to update it manually with an ALTER TABLE statement.")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _auto_sync_mysql_columns()
    print(f"Database tables ready (MySQL: {db_settings.name})")


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()