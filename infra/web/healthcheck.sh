#!/bin/sh
set -eu
# Nonzero status is recorded by systemd and can trigger an operations alarm.
curl --fail --silent --show-error --retry 5 --retry-connrefused --retry-delay 1 --max-time 10 http://127.0.0.1:8000/api/health > /dev/null
systemctl is-active --quiet equity-research.service
