"""Live dashboard for the AlphaZero bot.

Subscribes to telemetry events from uci.py and renders a browser dashboard:
chessboard, eval bar, MCTS depth / sim counter, game clock, time-since-go,
and a live-growing win-probability line plot.

Run separately from the UCI engine:

    uv run python monitor.py
    # then open http://localhost:8765 in your browser
    # start a game in your GUI / lichess-bot as usual

uci.py POSTs JSON events to /event when env var UCI_MONITOR_URL is set (or
its default http://localhost:8765/event). If the dashboard isn't running,
the POSTs fail-fast with no impact on the engine.

Event taxonomy (kind field):
    state       new game / board reset
    move        a move was just played (board update; win_prob if bot moved)
    tick        live MCTS update during search (cp, depth, sims, nps, top)
    go_start    bot received `go` command (clock, start timestamp)
    clock       clock update standalone
"""
import argparse
import json
import queue
import threading
import time
from typing import Iterator, Optional

import chess
import chess.svg
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

# --- Subscribers ---------------------------------------------------------
# Each browser tab connected to /stream has its own bounded queue. Slow tabs
# drop events at full-queue rather than blocking the engine emitter.
_subscribers_lock = threading.Lock()
_subscribers: list[queue.Queue] = []

# --- Latest snapshot -----------------------------------------------------
# Replayed to new subscribers so a browser refresh doesn't lose context.
_state_lock = threading.Lock()
_state = {
    "fen": chess.STARTING_FEN,
    "lastmove": None,
    "ply": 0,
    "bot_color": None,
    "win_prob_series": [],      # list of [ply, win_prob_bot_pov]  (MCTS Q)
    "nn_win_prob_series": [],   # list of [ply, win_prob_bot_pov]  (raw NN)
    "last_tick": None,
    "last_ponder_tick": None,   # latest ponder_tick (status=running); cleared on bot move / new game
    "clock": None,
    "board_svg": chess.svg.board(chess.Board(), size=400),
}


def _broadcast(event_json: str) -> None:
    with _subscribers_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(event_json)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def _render_board_svg(fen: str, lastmove_uci: Optional[str],
                      flipped: bool = False) -> str:
    try:
        board = chess.Board(fen)
    except Exception:
        return _state["board_svg"]
    last = None
    if lastmove_uci:
        try:
            last = chess.Move.from_uci(lastmove_uci)
        except Exception:
            last = None
    return chess.svg.board(board, lastmove=last, size=400, flipped=flipped)


def _bot_side_flipped() -> bool:
    """Board should be rendered black-at-bottom when the bot plays black."""
    return _state.get("bot_color") == "black"


@app.route("/event", methods=["POST"])
def event_in():
    payload = request.get_json(force=True, silent=True) or {}
    kind = payload.get("kind")
    extra_state_broadcast: Optional[dict] = None
    with _state_lock:
        if kind == "state":
            _state["fen"] = payload.get("fen", _state["fen"])
            _state["lastmove"] = payload.get("lastmove")
            _state["ply"] = payload.get("ply", 0)
            if payload.get("bot_color"):
                _state["bot_color"] = payload["bot_color"]
            # Fresh game -> reset plot history.
            if _state["ply"] == 0:
                _state["win_prob_series"] = []
                _state["nn_win_prob_series"] = []
                _state["last_tick"] = None
                _state["last_ponder_tick"] = None
            _state["board_svg"] = _render_board_svg(
                _state["fen"], _state["lastmove"], _bot_side_flipped()
            )
            payload["board_svg"] = _state["board_svg"]
            payload["flipped"] = _bot_side_flipped()
        elif kind == "move":
            _state["fen"] = payload.get("fen", _state["fen"])
            _state["lastmove"] = payload.get("lastmove")
            _state["ply"] = payload.get("ply", _state["ply"])
            _state["board_svg"] = _render_board_svg(
                _state["fen"], _state["lastmove"], _bot_side_flipped()
            )
            payload["board_svg"] = _state["board_svg"]
            payload["flipped"] = _bot_side_flipped()
            wp = payload.get("win_prob")
            ply = payload.get("ply")
            if wp is not None and ply is not None:
                _state["win_prob_series"].append([ply, wp])
            nn_wp = payload.get("nn_win_prob")
            if nn_wp is not None and ply is not None:
                _state["nn_win_prob_series"].append([ply, nn_wp])
            # Bot just moved -- any prior ponder state is stale.
            if payload.get("mover") == "bot":
                _state["last_ponder_tick"] = None
        elif kind == "tick":
            _state["last_tick"] = payload
        elif kind == "ponder_tick":
            # Keep the latest running tick. Ignore 'stopped' events: we want
            # the final count to remain visible across opp's think time and
            # our next search, until we move again (which clears it via the
            # mover=='bot' branch in the "move" handler above).
            if payload.get("status") != "stopped":
                _state["last_ponder_tick"] = payload
        elif kind == "go_start":
            if payload.get("clock"):
                _state["clock"] = payload["clock"]
            new_color = payload.get("bot_color")
            if new_color and new_color != _state.get("bot_color"):
                _state["bot_color"] = new_color
                # Bot color just became known/changed -- re-render the current
                # board with the right orientation and push a state update so
                # already-loaded dashboards flip immediately.
                _state["board_svg"] = _render_board_svg(
                    _state["fen"], _state["lastmove"], _bot_side_flipped()
                )
                extra_state_broadcast = {
                    "kind": "state",
                    "fen": _state["fen"],
                    "lastmove": _state["lastmove"],
                    "ply": _state["ply"],
                    "bot_color": _state["bot_color"],
                    "board_svg": _state["board_svg"],
                    "flipped": _bot_side_flipped(),
                }
        elif kind == "clock":
            _state["clock"] = {k: v for k, v in payload.items() if k != "kind"}
    _broadcast(json.dumps(payload))
    if extra_state_broadcast is not None:
        _broadcast(json.dumps(extra_state_broadcast))
    return jsonify({"ok": True})


@app.route("/stream")
def stream():
    q: queue.Queue = queue.Queue(maxsize=500)
    with _subscribers_lock:
        _subscribers.append(q)

    def gen() -> Iterator[bytes]:
        # Catch-up snapshot so a freshly-opened tab gets the current state.
        with _state_lock:
            snap = dict(_state)
            snap["kind"] = "snapshot"
        yield f"data: {json.dumps(snap)}\n\n".encode()
        try:
            while True:
                try:
                    data = q.get(timeout=15.0)
                    yield f"data: {data}\n\n".encode()
                except queue.Empty:
                    # SSE keepalive comment; some proxies close idle streams.
                    yield b": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _subscribers_lock:
                try:
                    _subscribers.remove(q)
                except ValueError:
                    pass

    return Response(gen(), mimetype="text/event-stream")


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>AlphaZero Bot Monitor</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<style>
  :root { color-scheme: dark; }
  body { font: 14px/1.4 system-ui, -apple-system, sans-serif;
         background: #161616; color: #ddd; margin: 0; padding: 20px; }
  h1 { font-size: 18px; margin: 0 0 16px; color: #fff; font-weight: 500; }
  .status { display: inline-block; padding: 2px 10px; border-radius: 4px;
            font-size: 11px; margin-left: 10px; vertical-align: middle; }
  .status.live { background: #1e7e3a; color: white; }
  .status.dead { background: #7e1e1e; color: white; }
  .grid { display: grid;
          grid-template-columns: min(calc(88vh + 70px), 870px) 1fr;
          gap: 24px; max-width: 1600px; }
  /* Right column fills the same vertical span as the board so the line
     plot stretches to fill whatever space the stats panel doesn't take. */
  .right-col { display: flex; flex-direction: column; gap: 16px;
               height: min(88vh, 800px); }
  .right-col #plot { flex: 1 1 auto; min-height: 0; }
  .board-area { display: flex; gap: 14px; align-items: flex-start; }
  #board { width: min(88vh, 800px); }
  /* chessboard.js square overrides for the dark theme */
  .white-1e1d7 { background-color: #ebecd0 !important; }
  .black-3c85d { background-color: #739552 !important; }
  /* Optional lastmove highlight (squares get .lastmove-from / .lastmove-to). */
  .lastmove-from { box-shadow: inset 0 0 0 3px #ffd54f80; }
  .lastmove-to   { box-shadow: inset 0 0 0 3px #ffd54f; }
  .eval-bar { width: 42px; height: min(88vh, 800px); background: #2a2a2a;
              border-radius: 4px; position: relative; overflow: hidden;
              border: 1px solid #3a3a3a; }
  .eval-fill { position: absolute; bottom: 0; left: 0; right: 0;
               background: linear-gradient(0deg, #2e7d32 0%, #66bb6a 100%);
               transition: height 200ms ease; height: 50%; }
  .meta { margin-top: 14px; color: #aaa; font-size: 12px;
          font-family: ui-monospace, monospace; }
  .stats { background: #1f1f1f; padding: 18px 18px 14px; border-radius: 8px;
           border: 1px solid #2a2a2a; }
  .stats .row { display: flex; justify-content: space-between;
                padding: 7px 0; border-bottom: 1px solid #2a2a2a; }
  .stats .row:last-child { border: none; }
  .stats .label { color: #888; font-size: 13px; }
  .stats .value { font-family: ui-monospace, monospace; font-size: 14px;
                  color: #ddd; }
  .stats .value.big { font-size: 22px; font-weight: 500; }
  .stats .value.eval-pos { color: #66bb6a; }
  .stats .value.eval-neg { color: #ef5350; }
  .top-moves { font-family: ui-monospace, monospace; font-size: 12px;
               color: #bbb; max-width: 320px; text-align: right;
               word-break: break-all; }
  #plot { background: #1f1f1f; border-radius: 8px; margin-top: 16px;
          padding: 8px; border: 1px solid #2a2a2a; }
</style>
</head>
<body>
<h1>AlphaZero Bot Monitor <span id="status" class="status dead">Disconnected</span></h1>

<div class="grid">
  <div>
    <div class="board-area">
      <div id="board"></div>
      <div class="eval-bar"><div id="evalFill" class="eval-fill"></div></div>
    </div>
    <div class="meta">
      <span id="ply">ply 0</span>
      &nbsp;·&nbsp; <span id="turn">white to move</span>
      &nbsp;·&nbsp; bot: <span id="botColor">—</span>
    </div>
  </div>

  <div class="right-col">
    <div class="stats">
      <div class="row"><span class="label">Bot eval (cp)</span>
        <span id="evalText" class="value big">+0.00</span></div>
      <div class="row"><span class="label">Win prob (bot)</span>
        <span id="winProb" class="value">50.0%</span></div>
      <div class="row"><span class="label">MCTS depth</span>
        <span id="depth" class="value">0</span></div>
      <div class="row"><span class="label">Sims done</span>
        <span id="sims" class="value">0</span></div>
      <div class="row"><span class="label">NPS</span>
        <span id="nps" class="value">0</span></div>
      <div class="row"><span class="label">Pondering</span>
        <span class="value"><span id="ponderSims">—</span>
          &nbsp;sims&nbsp;·&nbsp;<span id="ponderNps">—</span>&nbsp;nps</span></div>
      <div class="row"><span class="label">Move time</span>
        <span class="value"><span id="goTime">—</span>
          &nbsp;·&nbsp; <span id="lastMoveTime">—</span></span></div>
      <div class="row"><span class="label">Clock&nbsp;&nbsp;W&nbsp;/&nbsp;B</span>
        <span class="value"><span id="wclock">—</span>
          &nbsp;/&nbsp; <span id="bclock">—</span></span></div>
      <div class="row"><span class="label">Top moves</span>
        <span id="topMoves" class="top-moves">—</span></div>
    </div>
    <div id="plot"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

let goStartMs = null;     // ms timestamp when latest `go` arrived
let plotInitialised = false;

// --- Chess board (animated) -----------------------------------------------
const board = Chessboard('board', {
  position: 'start',
  showNotation: true,
  moveSpeed: 300,        // ms slide animation; 'slow'=200, 'fast'=100
  snapbackSpeed: 0,
  appearSpeed: 200,
  trashSpeed: 100,
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
});

function fenBoardPart(fen) {
  // chessboard.js accepts just the board part of a FEN, no side-to-move etc.
  return (fen || '').split(' ')[0] || 'start';
}

function setBoardFen(fen, animate) {
  if (!fen) return;
  board.position(fenBoardPart(fen), animate !== false);
}

function highlightLastmove(uci) {
  // Remove existing highlights, then apply to the two squares of `uci`.
  $('board').querySelectorAll('.lastmove-from, .lastmove-to').forEach(el => {
    el.classList.remove('lastmove-from', 'lastmove-to');
  });
  if (!uci || uci.length < 4) return;
  const from = uci.slice(0, 2), to = uci.slice(2, 4);
  const f = $('board').querySelector('.square-' + from);
  const t = $('board').querySelector('.square-' + to);
  if (f) f.classList.add('lastmove-from');
  if (t) t.classList.add('lastmove-to');
}

function setBoardOrientation(botColor) {
  const want = botColor === 'black' ? 'black' : 'white';
  if (board.orientation() !== want) board.orientation(want);
}

// Board size is derived from viewport height (88vh up to 800px). When the
// window resizes, ask chessboard.js to recompute square sizes from its
// container's current width.
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    board.resize();
    if (plotInitialised) Plotly.Plots.resize('plot');
  }, 80);
});

function initPlot(xs, ys, nnXs, nnYs) {
  Plotly.newPlot('plot', [
    {
      x: xs.length ? xs : [0],
      y: ys.length ? ys : [0.5],
      mode: 'lines+markers',
      line: { color: '#66bb6a', width: 2, shape: 'spline' },
      marker: { size: 6, color: '#66bb6a' },
      name: 'MCTS Q/N'
    },
    {
      x: nnXs.length ? nnXs : [0],
      y: nnYs.length ? nnYs : [0.5],
      mode: 'lines+markers',
      line: { color: '#42a5f5', width: 1.5, shape: 'spline', dash: 'dot' },
      marker: { size: 5, color: '#42a5f5' },
      name: 'NN'
    }
  ], {
    paper_bgcolor: '#1f1f1f', plot_bgcolor: '#1f1f1f',
    font: { color: '#bbb', size: 11 },
    xaxis: { title: 'ply', gridcolor: '#2a2a2a', zerolinecolor: '#3a3a3a' },
    yaxis: { title: 'P(bot wins)', range: [0, 1],
             gridcolor: '#2a2a2a', zerolinecolor: '#3a3a3a',
             tickformat: '.0%' },
    legend: { orientation: 'h', x: 0, y: 1.12, font: { size: 11 } },
    margin: { l: 55, r: 20, t: 30, b: 40 }
  }, { displayModeBar: false, responsive: true });
  plotInitialised = true;
}
initPlot([], [], [], []);

function fmtClock(ms) {
  if (ms == null) return '—';
  const sTot = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(sTot / 60);
  const s = sTot % 60;
  return `${m}:${s.toString().padStart(2,'0')}`;
}

function setEval(cp, winProb) {
  if (cp != null) {
    const sign = cp >= 0 ? '+' : '';
    $('evalText').textContent = sign + (cp/100).toFixed(2);
    $('evalText').className = 'value big ' + (cp >= 0 ? 'eval-pos' : 'eval-neg');
  }
  if (winProb != null) {
    $('winProb').textContent = (winProb * 100).toFixed(1) + '%';
    $('evalFill').style.height = (winProb * 100).toFixed(1) + '%';
  }
}

function applyTick(t) {
  setEval(t.cp, t.win_prob);
  if (t.depth !== undefined) $('depth').textContent = t.depth;
  if (t.sims !== undefined) $('sims').textContent = t.sims.toLocaleString();
  if (t.nps !== undefined) $('nps').textContent = t.nps.toLocaleString();
  if (t.top && t.top.length) {
    $('topMoves').textContent = t.top
      .map(m => `${m.uci}(N=${m.N})`)
      .join(' ');
  }
}

function applyMove(m) {
  if (m.fen) setBoardFen(m.fen, true);   // animate
  // chessboard.js redraws on .position(); apply highlight after a tick so the
  // new square divs are in place before we paint .lastmove-* classes.
  if (m.lastmove) setTimeout(() => highlightLastmove(m.lastmove), 320);
  if (m.ply !== undefined) $('ply').textContent = 'ply ' + m.ply;
  if (m.turn !== undefined) $('turn').textContent = m.turn + ' to move';
  if (m.duration !== undefined) {
    $('lastMoveTime').textContent = m.duration.toFixed(2) + 's';
  }
  if (m.win_prob !== undefined && m.ply !== undefined && plotInitialised) {
    Plotly.extendTraces('plot',
      { x: [[m.ply]], y: [[m.win_prob]] }, [0]);
    setEval(m.cp, m.win_prob);
  }
  if (m.nn_win_prob !== undefined && m.ply !== undefined && plotInitialised) {
    Plotly.extendTraces('plot',
      { x: [[m.ply]], y: [[m.nn_win_prob]] }, [1]);
  }
  if (m.mover === 'bot') {
    goStartMs = null;
    $('goTime').textContent = '—';
    // Our move just landed — wipe the previous round's ponder count so the
    // counter starts fresh when pondering kicks off for the next round.
    $('ponderSims').textContent = '—';
    $('ponderNps').textContent = '—';
  }
}

function applyState(s) {
  if (s.bot_color) setBoardOrientation(s.bot_color);
  if (s.fen) setBoardFen(s.fen, s.ply !== 0);  // don't animate on fresh game
  if (s.lastmove) setTimeout(() => highlightLastmove(s.lastmove), 320);
  if (s.ply !== undefined) $('ply').textContent = 'ply ' + s.ply;
  if (s.bot_color) $('botColor').textContent = s.bot_color;
  if (s.ply === 0) {
    highlightLastmove(null);
    initPlot([], [], [], []);
    setEval(0, 0.5);
    $('depth').textContent = '0';
    $('sims').textContent = '0';
    $('nps').textContent = '0';
    $('ponderSims').textContent = '—';
    $('ponderNps').textContent = '—';
    $('lastMoveTime').textContent = '—';
    $('topMoves').textContent = '—';
  }
}

function applyClock(c) {
  if (c.wtime !== undefined) $('wclock').textContent = fmtClock(c.wtime);
  if (c.btime !== undefined) $('bclock').textContent = fmtClock(c.btime);
}

function applySnapshot(s) {
  if (s.bot_color) setBoardOrientation(s.bot_color);
  // Don't animate the catch-up snapshot — it'd look like every piece is
  // sliding from start to its current square.
  if (s.fen) setBoardFen(s.fen, false);
  if (s.lastmove) setTimeout(() => highlightLastmove(s.lastmove), 50);
  if (s.ply !== undefined) $('ply').textContent = 'ply ' + s.ply;
  if (s.bot_color) $('botColor').textContent = s.bot_color;
  if (s.clock) applyClock(s.clock);
  const hasMcts = s.win_prob_series && s.win_prob_series.length > 0;
  const hasNn = s.nn_win_prob_series && s.nn_win_prob_series.length > 0;
  if (hasMcts || hasNn) {
    const xs  = hasMcts ? s.win_prob_series.map(p => p[0]) : [];
    const ys  = hasMcts ? s.win_prob_series.map(p => p[1]) : [];
    const nxs = hasNn   ? s.nn_win_prob_series.map(p => p[0]) : [];
    const nys = hasNn   ? s.nn_win_prob_series.map(p => p[1]) : [];
    initPlot(xs, ys, nxs, nys);
    if (hasMcts) setEval(null, ys[ys.length - 1]);
  }
  if (s.last_tick) applyTick(s.last_tick);
  if (s.last_ponder_tick && s.last_ponder_tick.status !== 'stopped') {
    $('ponderSims').textContent = s.last_ponder_tick.sims.toLocaleString();
    $('ponderNps').textContent = s.last_ponder_tick.nps.toLocaleString();
  } else {
    $('ponderSims').textContent = '—';
    $('ponderNps').textContent = '—';
  }
}

const es = new EventSource('/stream');
es.onopen = () => {
  $('status').className = 'status live';
  $('status').textContent = 'Live';
};
es.onerror = () => {
  $('status').className = 'status dead';
  $('status').textContent = 'Reconnecting…';
};
es.onmessage = e => {
  let ev;
  try { ev = JSON.parse(e.data); } catch { return; }
  switch (ev.kind) {
    case 'tick':     applyTick(ev); break;
    case 'move':     applyMove(ev); break;
    case 'state':    applyState(ev); break;
    case 'clock':    applyClock(ev); break;
    case 'ponder_tick':
      // Only update on 'running' ticks. 'stopped' is advisory — leave the
      // final count visible so the user can see how much was pondered during
      // opp's think time. The display is reset later in applyMove when *we*
      // move again (mover === 'bot').
      if (ev.status === 'running') {
        $('ponderSims').textContent = ev.sims.toLocaleString();
        $('ponderNps').textContent = ev.nps.toLocaleString();
      }
      break;
    case 'go_start':
      goStartMs = Date.now();
      if (ev.clock) applyClock(ev.clock);
      if (ev.bot_color) $('botColor').textContent = ev.bot_color;
      break;
    case 'snapshot': applySnapshot(ev); break;
  }
};

// Local 100ms tick to update the "time on this move" counter smoothly.
setInterval(() => {
  if (goStartMs != null) {
    $('goTime').textContent = ((Date.now() - goStartMs) / 1000).toFixed(1) + 's';
  }
}, 100);
</script>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description="AlphaZero bot live monitor")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"Dashboard: http://{args.host}:{args.port}")
    print("Engine should POST events to "
          f"http://{args.host}:{args.port}/event (default UCI_MONITOR_URL).")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
