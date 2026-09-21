#!/usr/bin/env bash
# Run a long job (encoder bake-off, embedding pass, LoRA export, official cold run) OUTSIDE the tool-call timeout.
#   scripts/run_detached.sh <job-name> <command...>      -> returns immediately; state lives in runs/jobs/<job-name>/
#   scripts/job_status.sh [job-name]                     -> running | finished(exit=N), elapsed, log tail
# Rules: jobs must be resumable where possible (write progress to disk); the OFFICIAL cold run is the exception - it must
# complete uninterrupted so that evaluation_time is honest (docs/spec/01 D17). Never put secrets on the command line.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <job-name> <command...>" >&2; exit 64; }
name="$1"; shift
case "$name" in *[!A-Za-z0-9._-]*|"") echo "job name must match [A-Za-z0-9._-]+" >&2; exit 64;; esac
dir="runs/jobs/$name"; mkdir -p "$dir"
if [ -s "$dir/pid" ] && kill -0 "$(cat "$dir/pid")" 2>/dev/null; then echo "job '$name' already running (pid $(cat "$dir/pid"))" >&2; exit 1; fi
rm -f "$dir/exit_code" "$dir/finished"
printf '%s\n' "$*" > "$dir/cmd"; date -u +%FT%TZ > "$dir/started"
quoted=$(printf '%q ' "$@")
cat > "$dir/run.sh" <<RUN
#!/usr/bin/env bash
cd "$(pwd)"
$quoted > "$dir/log" 2>&1
echo \$? > "$dir/exit_code"
date -u +%FT%TZ > "$dir/finished"
RUN
chmod +x "$dir/run.sh"
if command -v setsid >/dev/null 2>&1; then nohup setsid "$dir/run.sh" >/dev/null 2>&1 </dev/null & else nohup "$dir/run.sh" >/dev/null 2>&1 </dev/null & fi
echo $! > "$dir/pid"
echo "started '$name' (pid $(cat "$dir/pid")); follow with: scripts/job_status.sh $name"
