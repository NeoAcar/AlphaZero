"""Persistent engine daemon.

Spawns uci.py once (model load + torch.compile + MCTS pre-warm: ~20s), then
serves multiple client sessions over a Unix socket. Each session looks like a
fresh UCI engine to lichess-bot, but the heavy startup cost is amortised
across every game played for the lifetime of this process.

Lifecycle:
  - First-ever invocation: spawns the engine subprocess, listens on the socket.
  - Each client connects, runs a UCI session, then disconnects (or sends `quit`).
  - On disconnect we send `ucinewgame` to the engine to reset state -- the
    engine subprocess itself is NEVER killed across sessions. Next client gets
    a fully-warmed engine.
  - The server exits on SIGTERM/SIGINT (then sends real `quit` to the engine)
    or if the engine subprocess dies (the wrapper script will re-spawn the
    server next time lichess-bot connects).

Run via `alphazero_uci.sh` which auto-starts this daemon when the socket is
missing. PID + socket live in /tmp; engine stderr goes to /tmp/alphazero_uci.log.
"""
import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

SOCKET_PATH = "/tmp/alphazero_uci.sock"
PID_PATH = "/tmp/alphazero_uci.pid"
LOG_PATH = "/tmp/alphazero_uci.log"


def already_running() -> bool:
    """Return True if a previous server instance is still alive."""
    try:
        pid = int(Path(PID_PATH).read_text().strip())
    except (FileNotFoundError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="uci.py", help="engine script to run")
    ap.add_argument(
        "--workdir",
        default=str(Path(__file__).resolve().parent),
        help="cwd for the engine subprocess",
    )
    args = ap.parse_args()

    if already_running():
        print(f"engine_server already running (pid in {PID_PATH})", file=sys.stderr)
        sys.exit(1)

    # Touch pid file early so concurrent spawns lose the race.
    Path(PID_PATH).write_text(str(os.getpid()))

    log_fh = open(LOG_PATH, "a", buffering=1)
    log_fh.write(f"\n[server] starting (pid={os.getpid()})\n")

    # Use the project venv's python so torch etc. resolve.
    workdir = Path(args.workdir)
    venv_py = workdir / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else sys.executable
    engine_cmd = [py, "-u", args.engine]  # -u: unbuffered I/O on engine side too
    log_fh.write(f"[server] spawning engine: {engine_cmd} cwd={workdir}\n")

    engine_proc = subprocess.Popen(
        engine_cmd,
        cwd=str(workdir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log_fh,
        bufsize=0,
    )
    log_fh.write(f"[server] engine spawned (pid={engine_proc.pid})\n")

    # Single shared lock across cleanup hooks.
    shutting_down = threading.Event()

    def cleanup():
        if shutting_down.is_set():
            return
        shutting_down.set()
        log_fh.write("[server] shutting down\n")
        try:
            if engine_proc.poll() is None:
                engine_proc.stdin.write(b"quit\n")
                engine_proc.stdin.flush()
                engine_proc.wait(timeout=3)
        except Exception:
            try:
                engine_proc.kill()
            except Exception:
                pass
        for p in (SOCKET_PATH, PID_PATH):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        log_fh.write("[server] stopped\n")
        log_fh.close()

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    # Listen socket.
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    sock.listen(4)
    log_fh.write(f"[server] listening on {SOCKET_PATH}\n")

    # One session at a time. lichess-bot uses concurrency:1 anyway.
    while True:
        try:
            client, _ = sock.accept()
        except OSError:
            break  # socket closed
        if engine_proc.poll() is not None:
            log_fh.write("[server] engine subprocess died; exiting\n")
            try:
                client.close()
            except Exception:
                pass
            break
        handle_client(client, engine_proc, log_fh)


def handle_client(client_sock, engine_proc, log_fh):
    """Run one UCI session.

    Sync barrier: after the client half-closes its WR side (i.e., it's done
    sending commands but still listening for the final bestmove), we send
    our own `isready` to the engine and wait for the matching `readyok` to
    forward through. Then we close the client socket. This ensures the
    client sees every byte of engine output (including bestmove for an
    in-flight `go`) before disconnecting, AND that the engine is quiet
    before the next session starts.

    Counting subtlety: the client may itself have sent `isready` during the
    session. Each client `isready` produces a `readyok` that belongs to the
    client. Our barrier `readyok` is the (client_isready_count + 1)-th.
    """
    log_fh.write("[server] client connected\n")

    state_lock = threading.Lock()
    client_isready_count = 0
    barrier_sent = False
    readyoks_seen = 0

    def forward_in():
        """Client → engine.stdin. The client's `quit` is filtered so the
        engine subprocess keeps running. Returns on client EOF or `quit`."""
        nonlocal client_isready_count
        buf = b""
        try:
            while True:
                data = client_sock.recv(4096)
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    stripped = line.strip().lower()
                    if stripped == b"quit":
                        return
                    if stripped == b"isready":
                        with state_lock:
                            client_isready_count += 1
                    try:
                        engine_proc.stdin.write(line + b"\n")
                        engine_proc.stdin.flush()
                    except BrokenPipeError:
                        return
        except OSError:
            pass

    def engine_reader():
        """engine.stdout → client. After the barrier has been sent, exit on
        the (client_isready_count + 1)-th readyok -- that's the response to
        our own isready, after every prior readyok has been delivered."""
        nonlocal readyoks_seen
        while True:
            line = engine_proc.stdout.readline()
            if not line:
                return
            is_readyok = b"readyok" in line
            if is_readyok:
                with state_lock:
                    readyoks_seen += 1
            try:
                client_sock.sendall(line)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Client really gone; drain remaining output to our barrier
                # readyok internally so the next session starts clean.
                with state_lock:
                    need = client_isready_count + 1
                if is_readyok and readyoks_seen >= need:
                    return
                while True:
                    rest = engine_proc.stdout.readline()
                    if not rest:
                        return
                    if b"readyok" in rest:
                        with state_lock:
                            readyoks_seen += 1
                            if readyoks_seen >= need:
                                return
            if is_readyok:
                with state_lock:
                    if barrier_sent and readyoks_seen >= client_isready_count + 1:
                        return

    tin = threading.Thread(target=forward_in, daemon=True)
    tout = threading.Thread(target=engine_reader, daemon=True)
    tin.start()
    tout.start()

    # Wait for the client to finish writing (EOF or filtered `quit`).
    tin.join()

    # Barrier. `stop` cancels any active ponder thread; `isready` produces
    # the readyok we wait on. Queued ahead of these are all the client's
    # commands -- including any `go` -- so their output gets forwarded
    # first, in order.
    with state_lock:
        barrier_sent = True
    try:
        engine_proc.stdin.write(b"stop\nisready\n")
        engine_proc.stdin.flush()
    except Exception:
        pass
    tout.join(timeout=60.0)

    # Engine is now quiet. Safe to close the client socket -- the client's
    # socket_to_stdout thread will see EOF and the wrapper exits cleanly.
    try:
        client_sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        client_sock.close()
    except OSError:
        pass

    # Reset engine state for the next session.
    try:
        engine_proc.stdin.write(b"ucinewgame\n")
        engine_proc.stdin.flush()
    except Exception:
        pass

    log_fh.write("[server] client disconnected, engine drained + reset\n")


if __name__ == "__main__":
    main()
