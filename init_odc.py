from sqlalchemy import create_engine
from datacube.drivers.postgres._core import ensure_db, has_schema

# Adjust user/password/DB name if yours differ:
CONN = "postgresql://odcadmin:odcpass@localhost:5432/datacube"

engine = create_engine(CONN)

if not has_schema(engine):
    ensure_db(engine, with_permissions=False)
    print("✅ ODC index initialized (schema created without role/permission changes).")
else:
    print("ℹ️ Schema already exists; nothing to do.")
