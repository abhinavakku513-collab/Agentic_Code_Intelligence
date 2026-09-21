#!/usr/bin/env bash
# scripts/job_status.sh [job-name]   (no argument: list all jobs)
set -euo pipefail
show() {
  d="runs/jobs/$1"; [ -d "$d" ] || { echo "no such job: $1" >&2; return 1; }
  if [ -f "$d/exit_code" ]; then st="finished exit=$(cat "$d/exit_code")"
  elif [ -s "$d/pid" ] && kill -0 "$(cat "$d/pid")" 2>/dev/null; then st="running pid=$(cat "$d/pid")"
  else st="dead (no exit code recorded - killed or crashed)"; fi
  echo "== $1: $st | started $(cat "$d/started") | cmd: $(cat "$d/cmd")"
  [ -f "$d/finished" ] && echo "   finished $(cat "$d/finished")"
  [ -f "$d/log" ] && { echo "   --- last 5 log lines ---"; tail -n 5 "$d/log" | sed 's/^/   /'; }
}
if [ $# -ge 1 ]; then show "$1"; else for d in runs/jobs/*/; do [ -d "$d" ] && show "$(basename "$d")"; done; fi
