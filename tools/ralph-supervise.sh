#!/bin/bash
# Run `ralph run` until a deadline, restarting it by its terminal outcome:
# ralph exits 0 for every done{*} and 1 for every failed{*}. Run by
# tools/ralph-angr.service. Exits 0 on a deliberate stop (deadline, queue_empty,
# idle, budget), 1 on a crash (systemd restarts it), 2 on auth, 3 on repeated
# runner_terminal. Delete .ralph/state/supervise-deadline to start a new window.
set -u
cd "$(dirname "$0")/.."
state=.ralph/state
[ -f $state/supervise-deadline ] || date -d "+${RALPH_DAYS:-7} days" +%s > $state/supervise-deadline
deadline=$(<$state/supervise-deadline)

# ralph's systemd-run --user scopes need the user bus, which a system service lacks.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}

log() { echo "$(date -Is) ralph-supervise: $*" | tee -a $state/supervise.log; }

retries=0
while (( $(date +%s) < deadline )); do
    ralph run --no-tui 2>&1 | tee $state/supervise-last-run.out
    outcome=$(grep -oE '(done|failed)\{[a-z_]+\}' $state/supervise-last-run.out | tail -1)
    log "ralph run ended: ${outcome:-no outcome}"
    case $outcome in
        'done{iter_cap}') retries=0 ;;
        'failed{runner_terminal}')
            (( ++retries > ${RALPH_MAX_RETRIES:-6} )) && exit 3
            sleep "${RALPH_RETRY_SECS:-900}" ;;
        'failed{auth}') log "log claude in again, then restart ralph-angr"; exit 2 ;;
        done* | 'failed{budget}') exit 0 ;;
        *) exit 1 ;;
    esac
done
log "deadline reached"
