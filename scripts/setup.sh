#!/usr/bin/env bash
#
# setup.sh — provision the anomdec management DB + account, install the tools,
#            and run a health test.  Idempotent: safe to re-run.
#
# Steps (toggle with flags):
#   1. install tools     (uv venv + editable install; falls back to venv+pip)
#   2. create DB role + database(s) + grants   (needs a PostgreSQL superuser)
#   3. write secret.example.yml -> your secret file (optional)
#   4. health test       (connectivity, CREATE/ALTER privilege, store stack,
#                         rescued-column migration, CLI entrypoints)
#
# Configuration via env (all have sensible defaults):
#   ANOMDEC_DB_HOST      (localhost)
#   ANOMDEC_DB_PORT      (5432)
#   ANOMDEC_DB_NAME      (anomdec)
#   ANOMDEC_DB_USER      (anomdec)
#   ANOMDEC_DB_PASSWORD  (anomdec_pass)        <- change for production!
#   PG_SUPERUSER         (postgres)            superuser used to create role/db
#   PG_SUPERPASSWORD     ()                     used only with --super-mode psql
#
# Flags:
#   --check-only         run only the health test (no install, no DB creation)
#   --no-install         skip the tool install step
#   --with-ui            also install the UI extra (dash/plotly)
#   --with-test-db       also create anomdec_test (for integration tests)
#   --super-mode MODE    sudo (default) | psql   how to reach the superuser
#   --run-tests          run the unit test suite as part of the health test
#   --write-secret PATH  render secret.example.yml -> PATH (fills DB password)
#   -h | --help
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults -------------------------------------------------------------
DB_HOST="${ANOMDEC_DB_HOST:-localhost}"
DB_PORT="${ANOMDEC_DB_PORT:-5432}"
DB_NAME="${ANOMDEC_DB_NAME:-anomdec}"
DB_USER="${ANOMDEC_DB_USER:-anomdec}"
DB_PASSWORD="${ANOMDEC_DB_PASSWORD:-anomdec_pass}"
PG_SUPERUSER="${PG_SUPERUSER:-postgres}"
PG_SUPERPASSWORD="${PG_SUPERPASSWORD:-}"

DO_INSTALL=1
DO_CREATE_DB=1
DO_HEALTH=1
WITH_UI=0
WITH_TEST_DB=0
SUPER_MODE="sudo"
RUN_TESTS=0
WRITE_SECRET=""

# --- args -----------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --check-only)    DO_INSTALL=0; DO_CREATE_DB=0 ;;
    --no-install)    DO_INSTALL=0 ;;
    --with-ui)       WITH_UI=1 ;;
    --with-test-db)  WITH_TEST_DB=1 ;;
    --super-mode)    SUPER_MODE="${2:?--super-mode needs an argument}"; shift ;;
    --run-tests)     RUN_TESTS=1 ;;
    --write-secret)  WRITE_SECRET="${2:?--write-secret needs a path}"; shift ;;
    -h|--help)       awk 'NR==1 && /^#!/ {next} /^#/ {sub(/^#( )?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

log()  { printf '\n=== %s ===\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

# --- superuser psql wrapper ----------------------------------------------
run_super() {  # extra psql args via "$@"; SQL on stdin
  if [ "$SUPER_MODE" = "sudo" ]; then
    sudo -u "$PG_SUPERUSER" psql -v ON_ERROR_STOP=1 "$@"
  else
    PGPASSWORD="$PG_SUPERPASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" \
      -U "$PG_SUPERUSER" -v ON_ERROR_STOP=1 "$@"
  fi
}

# --- step 1: install tools -----------------------------------------------
install_tools() {
  log "Installing tools"
  local extras="[dev]"
  [ "$WITH_UI" = "1" ] && extras="[dev,ui]"
  if command -v uv >/dev/null 2>&1; then
    info "using uv"
    uv venv || true
    uv pip install -e ".${extras}"
  else
    info "uv not found; using python3 venv + pip"
    command -v python3 >/dev/null || die "python3 not found"
    [ -d .venv ] || python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip >/dev/null
    ./.venv/bin/pip install -e ".${extras}"
  fi
  PY=".venv/bin/python"
  info "installed (extras=${extras})"
}

# --- step 2: create role + database(s) -----------------------------------
create_role() {
  run_super -d postgres >/dev/null <<SQL
DO \$do\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
    CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASSWORD}';
  ELSE
    ALTER ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASSWORD}';
  END IF;
END
\$do\$;
SQL
  info "role '${DB_USER}' ready"
}

create_one_db() {  # $1 = database name
  local db="$1"
  run_super -d postgres >/dev/null <<SQL
SELECT 'CREATE DATABASE ${db} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${db}')\gexec
SQL
  run_super -d "${db}" >/dev/null <<SQL
ALTER DATABASE ${db} OWNER TO ${DB_USER};
GRANT ALL PRIVILEGES ON DATABASE ${db} TO ${DB_USER};
GRANT ALL ON SCHEMA public TO ${DB_USER};
SQL
  info "database '${db}' ready (owner=${DB_USER}, schema grants applied)"
}

create_db() {
  log "Creating role + database (superuser=${PG_SUPERUSER}, mode=${SUPER_MODE})"
  command -v psql >/dev/null || die "psql client not found (install postgresql-client)"
  create_role
  create_one_db "${DB_NAME}"
  [ "$WITH_TEST_DB" = "1" ] && create_one_db "anomdec_test"
}

# --- step 3: secret file --------------------------------------------------
write_secret() {
  log "Writing secret file -> ${WRITE_SECRET}"
  [ -f secret.example.yml ] || die "secret.example.yml not found in repo root"
  if [ -f "$WRITE_SECRET" ]; then
    info "exists already; not overwriting ${WRITE_SECRET}"
    return
  fi
  sed "s/^ADM_DB_PASSWORD:.*/ADM_DB_PASSWORD: \"${DB_PASSWORD}\"/; \
       s/^ADM_DB_HOST:.*/ADM_DB_HOST: \"${DB_HOST}\"/" \
      secret.example.yml > "$WRITE_SECRET"
  chmod 600 "$WRITE_SECRET"
  info "wrote ${WRITE_SECRET} (chmod 600). Point ANOMDEC_SECRET_PATH at it."
}

# --- step 4: health test --------------------------------------------------
health_test() {
  log "Health test"
  export ANOMDEC_DB_HOST="$DB_HOST" ANOMDEC_DB_PORT="$DB_PORT" \
         ANOMDEC_DB_NAME="$DB_NAME" ANOMDEC_DB_USER="$DB_USER" \
         ANOMDEC_DB_PASSWORD="$DB_PASSWORD"

  "$PY" "${REPO_ROOT}/scripts/healthcheck.py" || die "admdb health test failed"

  info "checking CLI entrypoints"
  for cli in anomdec-detect anomdec-detect-fast anomdec-update-stats; do
    if [ -x ".venv/bin/${cli}" ]; then
      ".venv/bin/${cli}" --help >/dev/null 2>&1 && info "  [OK]   ${cli}" \
        || die "${cli} present but --help failed"
    else
      info "  [skip] ${cli} (entrypoint not installed; run without --no-install)"
    fi
  done

  if [ "$RUN_TESTS" = "1" ]; then
    info "running unit test suite"
    "$PY" -m pytest tests/unit -q || die "unit tests failed"
  fi
  log "ALL CHECKS PASSED"
}

# --- run ------------------------------------------------------------------
[ "$DO_INSTALL"   = "1" ] && install_tools
[ "$DO_CREATE_DB" = "1" ] && create_db
[ -n "$WRITE_SECRET" ]    && write_secret
[ "$DO_HEALTH"    = "1" ] && health_test
