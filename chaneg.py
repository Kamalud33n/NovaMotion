import argparse
import sys

from sqlalchemy import MetaData, create_engine, func, select

from database import Base, engine, init_db
import models  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description="One-time copy of an old SQLite database into MySQL.")
    parser.add_argument("sqlite_file", help="path to the old SQLite file")
    args = parser.parse_args()

    init_db()

    src_engine = create_engine(f"sqlite:///file:{args.sqlite_file}?mode=ro&uri=true")
    src_meta = MetaData()
    src_meta.reflect(bind=src_engine)

    with engine.connect() as dst:
        filled = [t.name for t in Base.metadata.sorted_tables
                  if dst.execute(select(func.count()).select_from(t)).scalar()]
    if filled:
        print(f"Aborted: MySQL tables already contain data: {', '.join(filled)}")
        return 1

    copied = {}
    with src_engine.connect() as src, engine.begin() as dst:
        for table in Base.metadata.sorted_tables:
            if table.name not in src_meta.tables:
                continue
            src_table = src_meta.tables[table.name]
            usable = {c.name for c in table.columns} & {c.name for c in src_table.columns}
            rows = [{k: v for k, v in r._mapping.items() if k in usable}
                    for r in src.execute(select(src_table))]
            if rows:
                dst.execute(table.insert(), rows)
            copied[table.name] = len(rows)

    for name, n in copied.items():
        print(f"{name}: {n} rows copied")
    return 0


if __name__ == "__main__":
    sys.exit(main())