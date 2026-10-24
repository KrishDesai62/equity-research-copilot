#!/bin/sh
# Run as ubuntu with DATABASE_URL, LEDGER_BACKUP_BUCKET and AWS instance role.
set -eu
: "${DATABASE_URL:?required}" "${LEDGER_BACKUP_BUCKET:?required}"
umask 077
backup_dir=$(mktemp -d)
trap 'rm -rf "$backup_dir"' EXIT
stamp=$(date -u +%Y%m%dT%H%M%SZ)
pg_dump --dbname="$DATABASE_URL" --format=custom --file="$backup_dir/accounts.dump"
tar -czf "$backup_dir/imports.tar.gz" -C /home/ubuntu/equity-research-copilot/data/processed company_imports
aws s3 cp "$backup_dir/accounts.dump" "s3://$LEDGER_BACKUP_BUCKET/$stamp/accounts.dump" --sse AES256
aws s3 cp "$backup_dir/imports.tar.gz" "s3://$LEDGER_BACKUP_BUCKET/$stamp/imports.tar.gz" --sse AES256
