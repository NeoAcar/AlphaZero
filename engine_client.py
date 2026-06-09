"""Thin UCI client. Forwards stdin/stdout between lichess-bot and the
persistent engine server's Unix socket. See engine_server.py for the
rationale (model + JIT compile happen once across all games).

This script is invoked by alphazero_uci.sh and should look indistinguishable
from a real UCI engine to whoever spawned it.
"""
import os
import socket
import sys
import threading

SOCKET_PATH = "/tmp/alphazero_uci.sock"


def main() -> int:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"engine_client: cannot connect to {SOCKET_PATH}: {e}", file=sys.stderr)
        return 1

    stdin_done = threading.Event()

    def stdin_to_socket():
        try:
            while True:
                # Use readline so we forward complete UCI commands and don't
                # block the engine on partial-line buffering.
                data = sys.stdin.buffer.readline()
                if not data:
                    break
                try:
                    s.sendall(data)
                except (BrokenPipeError, OSError):
                    break
        finally:
            stdin_done.set()
            # Half-close so the server knows no more input is coming.
            try:
                s.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def socket_to_stdout():
        try:
            while True:
                data = s.recv(4096)
                if not data:
                    return
                try:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                except (BrokenPipeError, OSError):
                    return
        except OSError:
            return

    t = threading.Thread(target=stdin_to_socket, daemon=True)
    t.start()
    socket_to_stdout()
    return 0


if __name__ == "__main__":
    rc = main()
    # The stdin-forwarding daemon thread may be blocked in sys.stdin.readline()
    # at this point. Python's interpreter shutdown can't grab the stdin lock
    # while the daemon owns it -> "Fatal Python error: _enter_buffered_busy".
    # os._exit skips Python finalisation and lets the OS reap the thread,
    # which is the correct behaviour for a thin I/O forwarder anyway.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
