#!/bin/bash
# Stop the persistent AlphaZero engine daemon (engine_server.py + its warm
# uci.py subprocess) so the NEXT lichess-bot / chess-GUI launch via
# alphazero_uci.sh spawns a fresh engine that picks up your code edits.
#
# Why this is needed: the daemon is intentionally long-lived -- it keeps the
# model loaded + torch.compile'd across games to avoid the ~20s warmup every
# game. That means it holds the OLD uci.py / alphazero code in memory until it
# is restarted. Run this after editing engine code; then start lichess-bot.
#
#   ./restart_engine.sh
SOCKET=/tmp/alphazero_uci.sock
PID=/tmp/alphazero_uci.pid

# Verify a pid is actually OUR daemon before signalling it. The pid file can go
# stale (daemon crashed) and the OS may reuse that pid for an unrelated process
# -- blindly SIGTERMing it would kill an innocent bystander. Match the cmdline.
is_engine_daemon() {
    local pid=$1
    [ -r "/proc/$pid/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "engine_server.py"
}

if [ -f "$PID" ]; then
    pid=$(cat "$PID" 2>/dev/null)
    if [ -n "$pid" ] && is_engine_daemon "$pid"; then
        echo "stopping engine daemon (pid $pid)..."
        kill "$pid" 2>/dev/null          # SIGTERM -> daemon sends `quit` to the engine
        for _ in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.3
        done
        is_engine_daemon "$pid" && kill -9 "$pid" 2>/dev/null  # force if it ignored SIGTERM
    elif [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "pid $pid is alive but is NOT engine_server (stale/reused pid) -- not killing it"
    else
        echo "pid file present but process not alive (stale)"
    fi
else
    echo "no pid file; daemon not tracked"
fi

# Name-based sweep -- safe (can't hit a reused pid) and catches a crashed/orphaned
# daemon the pid file never knew about. This is the real workhorse.
pkill -f "engine_server.py" 2>/dev/null
pkill -f -- "-u uci.py" 2>/dev/null     # matches the daemon's `python -u uci.py`, not editors

rm -f "$SOCKET" "$PID"
echo "done -- daemon stopped, socket+pid cleared. Next launch is fresh."
