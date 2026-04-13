import os
import time
from urllib.parse import urlparse

import psycopg2


def main() -> None:
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    parsed = urlparse(database_url)
    host = parsed.hostname or "postgres"
    port = parsed.port or 5432
    user = parsed.username or ""
    dbname = (parsed.path or "").lstrip("/") or "postgres"

    max_attempts = int(os.getenv("DB_WAIT_ATTEMPTS", "30"))
    delay_seconds = int(os.getenv("DB_WAIT_DELAY_SECONDS", "2"))

    for attempt in range(1, max_attempts + 1):
        try:
            conn = psycopg2.connect(database_url)
            conn.close()
            print(f"PostgreSQL is ready at {host}:{port} (db={dbname}, user={user})")
            return
        except psycopg2.OperationalError as exc:
            print(f"Waiting for PostgreSQL ({attempt}/{max_attempts}): {exc}")
            time.sleep(delay_seconds)

    raise RuntimeError("PostgreSQL did not become ready in time")


if __name__ == "__main__":
    main()
