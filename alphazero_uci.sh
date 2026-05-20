#!/bin/bash
# UCI launcher. Routes lichess-bot (or any UCI host) to a long-lived engine
# daemon so the ~20s model load + torch.compile + MCTS pre-warm only happens
# ONCE -- not on every game. See engine_server.py.

set -e
DIR=/home/neo/PythonProjects/AlphaZero
cd "$DIR"

SOCKET=/tmp/alphazero_uci.sock
PID=/tmp/alphazero_uci.pid
PY="$DIR/.venv/bin/python3.12"

server_alive() {
    [ -S "$SOCKET" ] && [ -f "$PID" ] && kill -0 "$(cat "$PID" 2>/dev/null)" 2>/dev/null
}

if ! server_alive; then
    # Stale state left over from a crashed daemon -- clean up so the new one can bind.
    rm -f "$SOCKET" "$PID"
    # Detach the server: nohup + setsid so it survives this shell exiting.
    nohup setsid "$PY" -u "$DIR/engine_server.py" \
        > /tmp/alphazero_uci_server.out 2>&1 < /dev/null &
    # Wait for the socket to appear (server creates it before listening).
    # Capped at 90s in case the JIT cache is cold (first run on a machine).
    for _ in $(seq 1 90); do
        [ -S "$SOCKET" ] && break
        sleep 1
    done
fi

exec "$PY" -u "$DIR/engine_client.py"
