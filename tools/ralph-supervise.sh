#!/bin/bash
# Supervise `ralph run` for an unattended multi-day session; run by
# tools/ralph-angr.service (installed as /etc/systemd/system/ralph-angr.service).
#
# `ralph run` exits 0 on any done{*} and 1 on any failed{*}, so the exit code
# alone cannot tell "start the next run" from "stop". This reads the terminal
# outcome from the run's narrative instead:
#   done{iter_cap}           start the next run (max_iterations ended this one)
#   failed{runner_terminal}  wait RALPH_RETRY_SECS and retry, at most
#                            RALPH_MAX_RETRIES times in a row (exit 3 after)
#   done{queue_empty|idle}   stop: nothing left to do (exit 0)
#   failed{budget}           stop (exit 0)
#   failed{auth}             stop and wait for a human to log claude in (exit 2)
#   no outcome               ralph crashed or hit an infrastructure error;
#                            exit 1 so systemd restarts the service
# The service stops for good at the deadline in .ralph/state/supervise-deadline,
# written on first start as now + RALPH_DAYS days. Delete it to start a new window.
set -u
cd "$(dirname "$0")/.."
STATE=.ralph/state
LOG=$STATE/supervise.log
DEADLINE_FILE=$STATE/supervise-deadline
RETRY_SECS=${RALPH_RETRY_SECS:-900}
MAX_RETRIES=${RALPH_MAX_RETRIES:-6}

# ralph isolates each iteration with `systemd-run --user --scope`, which needs
# the user manager's bus; a system service does not inherit it.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}

log() { echo "$(date -Is) ralph-supervise: $*" | tee -a "$LOG"; }

[ -f "$DEADLINE_FILE" ] || date -d "+${RALPH_DAYS:-7} days" +%s > "$DEADLINE_FILE"
deadline=$(cat "$DEADLINE_FILE")
retries=0

while :; do
    if (( $(date +%s) >= deadline )); then
        log "deadline $(date -d "@$deadline" -Is) reached; stopping"
        exit 0
    fi
    out=$STATE/supervise-last-run.out
    ralph run --no-tui 2>&1 | tee "$out"
    rc=${PIPESTATUS[0]}
    outcome=$(grep -oE '(done|failed)\{[a-z_]+\}' "$out" | tail -1)
    log "ralph run exited rc=$rc outcome=${outcome:-none}"
    case $outcome in
        'done{iter_cap}')
            retries=0 ;;
        'failed{runner_terminal}')
            retries=$((retries + 1))
            if (( retries > MAX_RETRIES )); then
                log "runner_terminal $retries times in a row; stopping"
                exit 3
            fi
            sleep "$RETRY_SECS" ;;
        'done{queue_empty}' | 'done{idle}' | 'failed{budget}')
            exit 0 ;;
        'failed{auth}')
            log "auth failure: log claude in again, then: sudo systemctl restart ralph-angr"
            exit 2 ;;
        *)
            exit 1 ;;
    esac
done
