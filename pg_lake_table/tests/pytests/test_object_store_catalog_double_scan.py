from utils_pytest import *
import json
import threading

# A query that references the same read-only object_store table twice must see
# ONE Iceberg snapshot for both references.
#
# pg_lake resolves such a table per range table entry rather than per query.
# ReplacePgLakeTableWithReadTableFunc appends one RTE per occurrence, and
# CreatePgLakeScanSnapshot then calls CreateTableScanForRelation for each of
# them, which re-reads catalog.json through GetJsonFromURINoCache.  The
# Postgres transaction snapshot taken in CreatePgLakeScanSnapshot covers the
# data_files catalog, not an external Iceberg table, so a catalog.json rewrite
# that lands between the two reads hands the two references two different
# snapshots of the same table.
#
# Both tests below build two snapshots of one writable object_store table (an
# earlier metadata location and a later one), then rewrite catalog.json
# between them the way another cluster writing the same catalog would, and
# assert the invariants a single-snapshot read has to satisfy.


# Flip catalog.json between two precomputed bodies as fast as storage allows.
# The bodies are built up front so the thread only issues put_object, which
# keeps the rewrite far cheaper than the query it races.
class CatalogFlipper(threading.Thread):
    def __init__(self, s3, key, body_early, body_late):
        super().__init__(daemon=True)
        self.s3 = s3
        self.key = key
        self.bodies = [body_early, body_late]
        self.stop_event = threading.Event()
        self.flips = 0

    def run(self):
        while not self.stop_event.is_set():
            self.s3.put_object(
                Bucket=TEST_BUCKET, Key=self.key, Body=self.bodies[self.flips % 2]
            )
            self.flips += 1

    def stop(self):
        self.stop_event.set()
        self.join(timeout=30)


def _catalog_key(pg_conn, catalog_prefix="tmp"):
    dbname = run_query("SELECT current_database()", pg_conn)[0][0]
    return f"{catalog_prefix}/catalog/{dbname}/catalog.json"


def _current_metadata_location(pg_conn, table_name):
    return run_query(
        f"SELECT metadata_location FROM iceberg_tables "
        f"WHERE table_name = '{table_name}'",
        pg_conn,
    )[0][0]


# Build a catalog.json body whose entry for `table_name` points at `location`.
# The body is written straight to storage, so pg_lake never sees it coming.
def _catalog_body_pointing_at(s3, key, table_name, location):
    body = json.loads(s3.get_object(Bucket=TEST_BUCKET, Key=key)["Body"].read())
    for entry in body["tables"]:
        if entry["table-name"] == table_name:
            entry["metadata-location"] = location
    return json.dumps(body).encode("utf-8")


def _put_catalog(s3, key, body):
    s3.put_object(Bucket=TEST_BUCKET, Key=key, Body=body)


# Spread rows over single-row commits so each one lands in its own data file.
# The gap between the two catalog.json reads is the first reference's metadata
# and manifest read, so more manifest work widens the window.  Compaction is
# off on these tables, so the files stay separate and the metadata locations
# stay where the test pinned them.
def _insert_in_separate_commits(pg_conn, table, value_rows):
    for row in value_rows:
        run_command(f"INSERT INTO {table} VALUES {row}", pg_conn)
        pg_conn.commit()


def _create_uncompacted_writable_table(pg_conn, qualified_name, columns):
    # autovacuum_compact_data_files=false keeps the many small data files from
    # being rewritten into one, which would both narrow the window and move
    # the metadata location this test pins.
    run_command(
        f"CREATE TABLE {qualified_name}({columns}) USING iceberg "
        f"WITH (catalog='object_store', autovacuum_enabled = false, "
        f"autovacuum_compact_data_files = false)",
        pg_conn,
    )
    pg_conn.commit()


# Two references to the same table in one query must report the same count.
def test_double_reference_sees_one_snapshot(
    pg_conn, s3, extension, with_default_location, adjust_object_store_settings
):
    schema = "test_osc_double_scan"
    run_command(f"DROP SCHEMA IF EXISTS {schema} CASCADE", pg_conn)
    run_command(f"CREATE SCHEMA {schema}", pg_conn)
    pg_conn.commit()

    _create_uncompacted_writable_table(pg_conn, f"{schema}.wrt_tbl", "a INT")

    early_rows, extra_rows = 24, 12
    _insert_in_separate_commits(
        pg_conn, f"{schema}.wrt_tbl", [f"({n})" for n in range(early_rows)]
    )
    wait_until_object_store_writable_table_pushed(pg_conn, schema, "wrt_tbl")
    early_location = _current_metadata_location(pg_conn, "wrt_tbl")

    _insert_in_separate_commits(
        pg_conn,
        f"{schema}.wrt_tbl",
        [f"({n})" for n in range(early_rows, early_rows + extra_rows)],
    )
    wait_until_object_store_writable_table_pushed(pg_conn, schema, "wrt_tbl")
    late_location = _current_metadata_location(pg_conn, "wrt_tbl")
    assert early_location != late_location

    run_command(
        f"CREATE TABLE {schema}.read_tbl(a INT) USING iceberg "
        f"WITH (catalog='object_store', read_only=True, catalog_table_name='wrt_tbl')",
        pg_conn,
    )
    pg_conn.commit()

    key = _catalog_key(pg_conn)
    body_early = _catalog_body_pointing_at(s3, key, "wrt_tbl", early_location)
    body_late = _catalog_body_pointing_at(s3, key, "wrt_tbl", late_location)

    # Precondition: both snapshots are independently readable and differ, so a
    # count mismatch below really does mean two snapshots in one query.
    single_query = f"SELECT count(*) FROM {schema}.read_tbl"
    for body, expected in (
        (body_early, early_rows),
        (body_late, early_rows + extra_rows),
    ):
        _put_catalog(s3, key, body)
        observed = run_query(single_query, pg_conn)[0][0]
        pg_conn.commit()
        assert observed == expected, f"expected {expected} rows, got {observed}"

    doubled_query = (
        f"SELECT (SELECT count(*) FROM {schema}.read_tbl) AS first_reference, "
        f"       (SELECT count(*) FROM {schema}.read_tbl) AS second_reference"
    )

    flipper = CatalogFlipper(s3, key, body_early, body_late)
    flipper.start()
    mismatches = []
    attempts = 0
    try:
        for attempts in range(1, 201):
            rows = run_query(doubled_query, pg_conn, raise_error=False)
            if isinstance(rows, str):
                pg_conn.rollback()
                continue
            pg_conn.commit()
            first_reference, second_reference = rows[0][0], rows[0][1]
            if first_reference != second_reference:
                mismatches.append((attempts, first_reference, second_reference))
                break
    finally:
        flipper.stop()

    print(
        f"\ndouble-reference count check: {attempts} queries, "
        f"{flipper.flips} catalog.json rewrites, {len(mismatches)} mismatches"
    )
    for attempt, first_reference, second_reference in mismatches:
        print(
            f"  query #{attempt}: first_reference={first_reference} "
            f"second_reference={second_reference}"
        )

    assert not mismatches, (
        "two references to one read-only object_store table in a single query "
        "returned different row counts, so each resolved its own Iceberg "
        f"snapshot: {mismatches}"
    )


# The same defect through a view, where the two references are not obviously
# the same table to the reader of the query.  The view keeps the rows of a
# heap table that no later row supersedes, plus the added rows that no later
# row supersedes.  It names the read-only object_store table twice: once in
# the CTE, once in the second half of the UNION ALL.  Under one snapshot the
# view returns exactly one row per record key.
def test_view_referencing_one_table_twice(
    pg_conn, s3, extension, with_default_location, adjust_object_store_settings
):
    schema = "test_osc_view_twice"
    run_command(f"DROP SCHEMA IF EXISTS {schema} CASCADE", pg_conn)
    run_command(f"CREATE SCHEMA {schema}", pg_conn)
    pg_conn.commit()

    columns = "record_key INT, amount INT, kind TEXT, version INT"
    _create_uncompacted_writable_table(pg_conn, f"{schema}.wrt_rows", columns)

    # Padding rows carry a kind that neither half of the view selects, so they
    # only serve to widen the window between the two catalog.json reads.
    padding = [f"({1000 + n}, {n}, 'noop', {n})" for n in range(24)]
    batch_at_version5 = ["(1, 100, 'supersede', 5)", "(1, 101, 'add', 5)"]
    batch_at_version9 = ["(1, 101, 'supersede', 9)", "(1, 102, 'add', 9)"]

    _insert_in_separate_commits(
        pg_conn, f"{schema}.wrt_rows", padding + batch_at_version5
    )
    wait_until_object_store_writable_table_pushed(pg_conn, schema, "wrt_rows")
    early_location = _current_metadata_location(pg_conn, "wrt_rows")

    _insert_in_separate_commits(pg_conn, f"{schema}.wrt_rows", batch_at_version9)
    wait_until_object_store_writable_table_pushed(pg_conn, schema, "wrt_rows")
    late_location = _current_metadata_location(pg_conn, "wrt_rows")
    assert early_location != late_location

    run_command(
        f"CREATE TABLE {schema}.read_rows({columns}) USING iceberg "
        f"WITH (catalog='object_store', read_only=True, "
        f"catalog_table_name='wrt_rows')",
        pg_conn,
    )
    run_command(f"CREATE TABLE {schema}.base_tbl(record_key INT, amount INT)", pg_conn)
    run_command(f"INSERT INTO {schema}.base_tbl VALUES (1, 100)", pg_conn)
    pg_conn.commit()

    run_command(
        f"""CREATE VIEW {schema}.effective_rows AS
            WITH superseded AS (
                SELECT record_key, MAX(version) AS version
                FROM {schema}.read_rows
                WHERE kind = 'supersede'
                GROUP BY record_key
            )
            SELECT base.record_key, base.amount
            FROM {schema}.base_tbl base
            WHERE NOT EXISTS (
                SELECT 1 FROM superseded
                WHERE superseded.record_key = base.record_key
            )
            UNION ALL
            SELECT added.record_key, added.amount
            FROM {schema}.read_rows added
            WHERE added.kind = 'add'
              AND NOT EXISTS (
                  SELECT 1 FROM superseded
                  WHERE superseded.record_key = added.record_key
                    AND superseded.version > added.version
              )""",
        pg_conn,
    )
    pg_conn.commit()

    key = _catalog_key(pg_conn)
    body_early = _catalog_body_pointing_at(s3, key, "wrt_rows", early_location)
    body_late = _catalog_body_pointing_at(s3, key, "wrt_rows", late_location)

    view_query = (
        f"SELECT record_key, amount FROM {schema}.effective_rows ORDER BY amount"
    )

    # Precondition: each snapshot on its own gives exactly one row, and the two
    # differ, so anything else below comes from mixing the two.
    expected_results = []
    for body, expected in ((body_early, [(1, 101)]), (body_late, [(1, 102)])):
        _put_catalog(s3, key, body)
        rows = run_query(view_query, pg_conn)
        pg_conn.commit()
        observed = [tuple(row) for row in rows]
        assert observed == expected, f"expected {expected}, got {observed}"
        expected_results.append(expected)

    flipper = CatalogFlipper(s3, key, body_early, body_late)
    flipper.start()
    wrong_results = []
    attempts = 0
    try:
        for attempts in range(1, 201):
            rows = run_query(view_query, pg_conn, raise_error=False)
            if isinstance(rows, str):
                pg_conn.rollback()
                continue
            pg_conn.commit()
            observed = [tuple(row) for row in rows]
            if observed not in expected_results:
                wrong_results.append((attempts, observed))
                break
    finally:
        flipper.stop()

    print(
        f"\nview result check: {attempts} queries, {flipper.flips} "
        f"catalog.json rewrites, {len(wrong_results)} wrong results"
    )
    for attempt, observed in wrong_results:
        print(f"  query #{attempt}: {observed}")

    assert not wrong_results, (
        "a view naming one read-only object_store table twice returned rows "
        "that no single Iceberg snapshot of that table can produce, expected "
        f"one of {expected_results}: {wrong_results}"
    )
