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
    "nn_std_series": [],        # list of [ply, std_dev_of_nn_win_prob]  (WDL only)
    "moves_left_plies": None,   # ChessFormer auxiliary; display only
    "moves_left_alpha": None,
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
                _state["nn_std_series"] = []
                _state["moves_left_plies"] = None
                _state["moves_left_alpha"] = None
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
            nn_std = payload.get("nn_std")
            if nn_std is not None and ply is not None:
                _state["nn_std_series"].append([ply, nn_std])
            if payload.get("moves_left_plies") is not None:
                _state["moves_left_plies"] = payload["moves_left_plies"]
                _state["moves_left_alpha"] = payload.get("moves_left_alpha")
            # Bot just moved -- any prior ponder state is stale.
            if payload.get("mover") == "bot":
                _state["last_ponder_tick"] = None
        elif kind == "tick":
            _state["last_tick"] = payload
            if payload.get("moves_left_plies") is not None:
                _state["moves_left_plies"] = payload["moves_left_plies"]
                _state["moves_left_alpha"] = payload.get("moves_left_alpha")
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
  :root {
    color-scheme: dark;
    --bg: #121316; --panel: #1b1d22; --panel-2: #232630;
    --line: #2c2f38; --muted: #8b909b; --text: #e6e8ec;
    --accent: #6ea8fe; --pos: #5cc46a; --neg: #ef5350; --gold: #ffce54;
  }
  * { box-sizing: border-box; }
  body { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
         background: radial-gradient(1200px 700px at 80% -10%, #1a1d24 0%, var(--bg) 60%);
         color: var(--text); margin: 0; padding: 22px; }
  .hdr { display: flex; align-items: center; gap: 12px; margin: 0 0 18px; }
  h1 { font-size: 17px; margin: 0; color: #fff; font-weight: 600; letter-spacing: .2px; }
  h1 .az { color: var(--accent); }
  .status { display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px;
            border-radius: 999px; font-size: 11px; font-weight: 600; letter-spacing: .03em; }
  .status::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
  .status.live { background: rgba(92,196,106,.14); color: var(--pos); }
  .status.dead { background: rgba(239,83,80,.14); color: var(--neg); }
  .grid { display: grid;
          grid-template-columns: min(calc(88vh + 76px), 880px) 1fr;
          gap: 22px; max-width: 1640px; }
  .right-col { display: flex; flex-direction: column; gap: 16px;
               height: min(88vh, 820px); }
  .right-col #plot { flex: 1 1 auto; min-height: 180px; }
  .board-area { display: flex; gap: 14px; align-items: flex-start; }
  #board { width: min(88vh, 800px);
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 8px 30px rgba(0,0,0,.45); }
  .board-wrap { position: relative; width: min(88vh, 800px); }
  /* SVG overlay for candidate-move arrows; sits above the board, ignores clicks. */
  #arrows { position: absolute; left: 0; top: 0; pointer-events: none; z-index: 5; }
  .white-1e1d7 { background-color: #eef0d6 !important; }
  .black-3c85d { background-color: #6f8f4e !important; }
  .lastmove-from { box-shadow: inset 0 0 0 3px rgba(255,206,84,.5); }
  .lastmove-to   { box-shadow: inset 0 0 0 3px var(--gold); }
  /* Eval bar: bot win-prob fills from the bottom; centre line marks 50%. */
  .eval-bar { width: 40px; height: min(88vh, 800px); background: #20232b;
              border-radius: 6px; position: relative; overflow: hidden;
              border: 1px solid var(--line); }
  .eval-fill { position: absolute; bottom: 0; left: 0; right: 0; height: 50%;
               background: linear-gradient(0deg, #2e7d32, #6ee07e);
               transition: height 220ms ease; }
  .eval-bar .mid { position: absolute; left: 0; right: 0; top: 50%;
                   height: 1px; background: rgba(255,255,255,.22); }
  .eval-bar .pct { position: absolute; left: 0; right: 0; top: 6px; text-align: center;
                   font: 600 11px ui-monospace, monospace; color: rgba(255,255,255,.8);
                   text-shadow: 0 1px 2px rgba(0,0,0,.6); }
  .meta { margin-top: 14px; color: var(--muted); font-size: 12.5px;
          font-family: ui-monospace, monospace; }
  .meta b { color: var(--text); font-weight: 600; }
  .legend { margin-top: 6px; font-size: 11.5px; color: var(--muted); }
  .legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px;
                vertical-align: middle; margin-right: 4px; }
  .card { background: var(--panel); padding: 16px 16px 12px; border-radius: 10px;
          border: 1px solid var(--line); }
  /* Big eval header */
  .eval-head { display: flex; align-items: baseline; justify-content: space-between;
               margin-bottom: 12px; }
  .eval-side { display: flex; align-items: center; gap: 9px; }
  .eval-head .cp { font: 600 30px/1 ui-monospace, monospace; }
  .eval-head .wp { font: 600 18px/1 ui-monospace, monospace; color: var(--muted); }
  .mlh-badge { padding: 2px 7px; border-radius: 999px;
               border: 1px solid var(--line); background: var(--panel-2);
               color: #b8bdc8; font: 600 11px/1.4 ui-monospace, monospace;
               white-space: nowrap; }
  .eval-pos { color: var(--pos); } .eval-neg { color: var(--neg); }
  .cp.mate { color: var(--gold); }   /* proven forced mate (M7) */
  /* Stat tiles */
  .tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .tile { background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px;
          padding: 8px 10px; }
  .tile .k { color: var(--muted); font-size: 10.5px; text-transform: uppercase;
             letter-spacing: .06em; }
  .tile .v { font: 600 16px/1.2 ui-monospace, monospace; color: var(--text); margin-top: 2px; }
  .tile .v small { font-size: 11px; color: var(--muted); font-weight: 400; }
  .sect { color: var(--muted); font-size: 10.5px; letter-spacing: .08em;
          text-transform: uppercase; margin: 14px 0 7px; }
  /* Candidate move bars */
  .cand { display: grid; grid-template-columns: 56px 1fr 44px 50px;
          align-items: center; gap: 9px; padding: 3px 0;
          font-family: ui-monospace, monospace; font-size: 13px; }
  .cand-move { color: var(--text); font-weight: 600; }
  .cand-bar { height: 15px; background: #20232b; border-radius: 4px; overflow: hidden; }
  .cand-fill { display: block; height: 100%; border-radius: 4px;
               transition: width 200ms ease; }
  .cand-n { color: var(--muted); text-align: right; font-size: 12px; }
  .cand-q { text-align: right; font-weight: 600; }
  .cand-empty { color: var(--muted); font-family: ui-monospace, monospace; }
  .pv-line { font-family: ui-monospace, monospace; font-size: 12.5px;
             color: #9fc0e6; word-break: break-word; line-height: 1.55;
             background: var(--panel-2); border: 1px solid var(--line);
             border-radius: 8px; padding: 8px 10px; min-height: 20px; }
  #plot { background: var(--panel); border-radius: 10px; padding: 8px;
          border: 1px solid var(--line); }
</style>
</head>
<body>
<div class="hdr">
  <h1><span class="az">Alpha</span>Zero Monitor</h1>
  <span id="status" class="status dead">Disconnected</span>
</div>

<div class="grid">
  <div>
    <div class="board-area">
      <div class="board-wrap">
        <div id="board"></div>
        <svg id="arrows"></svg>
      </div>
      <div class="eval-bar">
        <div id="evalFill" class="eval-fill"></div>
        <div class="mid"></div>
        <div id="evalPct" class="pct">50%</div>
      </div>
    </div>
    <div class="meta">
      <b id="ply">ply 0</b> &nbsp;·&nbsp; <span id="turn">white to move</span>
      &nbsp;·&nbsp; bot plays <b id="botColor">—</b>
    </div>
    <div class="meta legend">
      <span class="sw" style="background:#5cc46a"></span>bot candidates (Q-coloured)
      &nbsp;·&nbsp;
      <span class="sw" style="background:#5b9bd5"></span>predicted opponent (while pondering)
    </div>
  </div>

  <div class="right-col">
    <div class="card">
      <div class="eval-head">
        <span id="evalText" class="cp">+0.00</span>
        <span class="eval-side">
          <span id="movesLeft" class="mlh-badge" hidden></span>
          <span id="winProb" class="wp">50.0%</span>
        </span>
      </div>
      <div class="tiles">
        <div class="tile"><div class="k">Depth</div><div class="v" id="depth">0</div></div>
        <div class="tile"><div class="k">Sims</div><div class="v" id="sims">0</div></div>
        <div class="tile"><div class="k">NPS</div><div class="v" id="nps">0</div></div>
        <div class="tile"><div class="k">Move time</div>
          <div class="v"><span id="goTime">—</span> <small id="lastMoveTime">—</small></div></div>
        <div class="tile"><div class="k">Ponder</div>
          <div class="v"><span id="ponderSims">—</span> <small><span id="ponderNps">—</span> nps</small></div></div>
        <div class="tile"><div class="k">Clock W / B</div>
          <div class="v"><span id="wclock">—</span> <small>/</small> <span id="bclock">—</span></div></div>
      </div>
      <div class="sect">Candidate moves &nbsp;<small style="text-transform:none;letter-spacing:0">(visits · Q)</small></div>
      <div id="candidates"><div class="cand-empty">—</div></div>
      <div class="sect">Principal variation</div>
      <div id="pv" class="pv-line">—</div>
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
  if (board.orientation() !== want) {
    board.orientation(want);
    setTimeout(() => drawArrows(lastTop, lastOpp), 0);   // re-map arrows to new orientation
  }
}

// Board size is derived from viewport height (88vh up to 800px). When the
// window resizes, ask chessboard.js to recompute square sizes from its
// container's current width.
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    board.resize();
    drawArrows(lastTop, lastOpp);
    if (plotInitialised) Plotly.Plots.resize('plot');
  }, 80);
});

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

function initPlot(xs, ys, nnXs, nnYs, nnLo, nnHi) {
  Plotly.newPlot('plot', [
    // Trace 0: MCTS Q/N (visible)
    {
      x: xs.length ? xs : [0],
      y: ys.length ? ys : [0.5],
      mode: 'lines+markers',
      line: { color: '#66bb6a', width: 2, shape: 'spline' },
      marker: { size: 6, color: '#66bb6a' },
      name: 'MCTS Q/N'
    },
    // Trace 1: NN lower bound (invisible line; fill anchor)
    {
      x: nnXs.length ? nnXs : [0],
      y: nnLo.length ? nnLo : [0.5],
      mode: 'lines',
      line: { color: 'transparent', shape: 'spline' },
      showlegend: false,
      hoverinfo: 'skip',
    },
    // Trace 2: NN upper bound + shaded fill down to trace 1 (the band)
    {
      x: nnXs.length ? nnXs : [0],
      y: nnHi.length ? nnHi : [0.5],
      mode: 'lines',
      line: { color: 'transparent', shape: 'spline' },
      fill: 'tonexty',
      fillcolor: 'rgba(66, 165, 245, 0.18)',
      name: 'NN ±σ',
      hoverinfo: 'skip',
    },
    // Trace 3: NN center line (visible)
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
initPlot([], [], [], [], [], []);

function fmtClock(ms) {
  if (ms == null) return '—';
  const sTot = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(sTot / 60);
  const s = sTot % 60;
  return `${m}:${s.toString().padStart(2,'0')}`;
}

function setEval(cp, winProb, mate) {
  if (mate != null) {
    // Proven forced mate: show "M7" (bot mates) or "-M7" (bot mated) instead of
    // a saturated centipawn value, and peg the eval bar full/empty.
    const txt = (mate > 0 ? 'M' : '-M') + Math.abs(mate);
    $('evalText').textContent = txt;
    $('evalText').className = 'cp ' + (mate > 0 ? 'mate' : 'eval-neg');
    const p = mate > 0 ? 100 : 0;
    $('evalFill').style.height = p + '%';
    $('evalPct').textContent = txt;
    $('winProb').textContent = (mate > 0 ? '100' : '0') + '%';
    return;
  }
  if (cp != null) {
    const sign = cp >= 0 ? '+' : '';
    $('evalText').textContent = sign + (cp/100).toFixed(2);
    $('evalText').className = 'cp ' + (cp >= 0 ? 'eval-pos' : 'eval-neg');
  }
  if (winProb != null) {
    const p = winProb * 100;
    $('winProb').textContent = p.toFixed(1) + '%';
    $('evalFill').style.height = p.toFixed(1) + '%';
    $('evalPct').textContent = Math.round(p) + '%';
  }
}

function setMovesLeft(plies, alpha) {
  const el = $('movesLeft');
  if (plies == null || !Number.isFinite(Number(plies))) {
    el.hidden = true;
    el.textContent = '';
    el.title = '';
    return;
  }
  const moves = Math.max(0, Number(plies)) / 2;
  el.textContent = `≈${moves.toFixed(1)} hamle`;
  el.hidden = false;
  el.title = 'Modelin tahmini kalan tam hamle sayısı (moves-left μ)';
  if (alpha != null && Number.isFinite(Number(alpha))) {
    el.title += ` · dağılım α=${Number(alpha).toFixed(3)}`;
  }
}

// Map Q in [-1,1] to a red -> yellow -> green hue for the candidate bars.
function qColor(q) {
  const h = clamp01((q + 1) / 2) * 130;   // 0 = red, 130 = green
  return `hsl(${h.toFixed(0)}, 60%, 46%)`;
}

// --- Candidate-move arrows on the board ----------------------------------
const SVGNS = 'http://www.w3.org/2000/svg';
let lastTop = [];     // latest candidates, so we can redraw on resize / flip
let lastOpp = false;  // are the current arrows predicted-opponent (vs bot) moves?
const OPP_COLOR = '#5b9bd5';   // cool blue: predicted opponent replies (vs warm Q-coloured bot arrows)

function squareCenter(square, sq, orient) {
  // square like "e2" -> pixel centre, accounting for board orientation.
  const file = square.charCodeAt(0) - 97;        // a=0..h=7
  const rank = parseInt(square[1], 10) - 1;       // rank1=0..rank8=7
  let col, row;
  if (orient === 'white') { col = file;     row = 7 - rank; }
  else                    { col = 7 - file; row = rank; }
  return { x: (col + 0.5) * sq, y: (row + 0.5) * sq };
}

function drawArrows(top, opp) {
  const svg = $('arrows');
  if (!svg) return;
  const size = $('board').clientWidth || 0;
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.innerHTML = '';
  if (!top || !top.length || !size) return;
  const sq = size / 8;
  const orient = board.orientation();
  const maxN = Math.max(...top.map(m => m.N || 0)) || 1;
  // Draw weakest first so the most-visited arrow ends up on top.
  const ordered = [...top].sort((a, b) => (a.N || 0) - (b.N || 0));
  for (const m of ordered) {
    if (!m.uci || m.uci.length < 4) continue;
    const from = squareCenter(m.uci.slice(0, 2), sq, orient);
    const to   = squareCenter(m.uci.slice(2, 4), sq, orient);
    const share = (m.N || 0) / maxN;             // relative to the best move
    const w  = sq * (0.10 + 0.16 * share);       // thickness scales with visits
    const op = 0.28 + 0.55 * share;              // opacity scales with visits
    // Bot's own moves: colour by Q (red->green). Predicted opponent replies:
    // a single cool blue, so they're clearly not the bot's choices.
    const col = opp ? OPP_COLOR : qColor(m.q != null ? m.q : 0);
    let dx = to.x - from.x, dy = to.y - from.y;
    const L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
    const head = w * 2.4;
    const tipX = to.x - ux * sq * 0.16, tipY = to.y - uy * sq * 0.16;  // tip inside target
    const baseX = tipX - ux * head, baseY = tipY - uy * head;
    const nx = -uy, ny = ux, hw = head * 0.6;    // perpendicular for the head
    const line = document.createElementNS(SVGNS, 'line');
    line.setAttribute('x1', from.x); line.setAttribute('y1', from.y);
    line.setAttribute('x2', baseX);  line.setAttribute('y2', baseY);
    line.setAttribute('stroke', col); line.setAttribute('stroke-width', w);
    line.setAttribute('stroke-linecap', 'round'); line.setAttribute('opacity', op);
    svg.appendChild(line);
    const poly = document.createElementNS(SVGNS, 'polygon');
    poly.setAttribute('points',
      `${tipX},${tipY} ${baseX + nx * hw},${baseY + ny * hw} ${baseX - nx * hw},${baseY - ny * hw}`);
    poly.setAttribute('fill', col); poly.setAttribute('opacity', op);
    svg.appendChild(poly);
  }
}

function renderCandidates(top) {
  const el = $('candidates');
  if (!top || !top.length) { el.innerHTML = '<div class="cand-empty">—</div>'; return; }
  const totalN = top.reduce((s, m) => s + (m.N || 0), 0) || 1;
  el.innerHTML = top.map(m => {
    const share = (m.N || 0) / totalN;
    const q = (m.q != null) ? m.q : 0;
    const qtxt = (q >= 0 ? '+' : '') + q.toFixed(2);
    const qcls = q >= 0 ? 'eval-pos' : 'eval-neg';
    return `<div class="cand">
      <span class="cand-move">${m.uci}</span>
      <span class="cand-bar"><span class="cand-fill"
        style="width:${(share*100).toFixed(1)}%;background:${qColor(q)}"></span></span>
      <span class="cand-n">${(m.N || 0).toLocaleString()}</span>
      <span class="cand-q ${qcls}">${qtxt}</span>
    </div>`;
  }).join('');
}

function renderPV(pv) {
  $('pv').textContent = (pv && pv.length) ? pv.join('  ') : '—';
}

function applyTick(t) {
  setEval(t.cp, t.win_prob, t.mate);
  if (t.moves_left_plies !== undefined) {
    setMovesLeft(t.moves_left_plies, t.moves_left_alpha);
  }
  if (t.depth !== undefined) $('depth').textContent = t.depth;
  if (t.sims !== undefined) $('sims').textContent = t.sims.toLocaleString();
  if (t.nps !== undefined) $('nps').textContent = t.nps.toLocaleString();
  if (t.top) renderCandidates(t.top);
  if (t.pv) renderPV(t.pv);
}

function applyMove(m) {
  // The position is about to change -- clear stale candidate arrows; the next
  // search's ticks (or ponder ticks) will draw fresh ones for the new position.
  lastTop = []; lastOpp = false; drawArrows([]);
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
    setEval(m.cp, m.win_prob, m.mate);
  }
  if (m.nn_win_prob !== undefined && m.ply !== undefined && plotInitialised) {
    // Extend NN lower/upper bound + center together so the band stays in sync.
    // If no std was sent (non-WDL head), collapse the band to zero-width.
    const sigma = (m.nn_std !== undefined) ? m.nn_std : 0;
    const lo = clamp01(m.nn_win_prob - sigma);
    const hi = clamp01(m.nn_win_prob + sigma);
    Plotly.extendTraces('plot', {
      x: [[m.ply], [m.ply], [m.ply]],
      y: [[lo], [hi], [m.nn_win_prob]],
    }, [1, 2, 3]);
  }
  if (m.moves_left_plies !== undefined) {
    setMovesLeft(m.moves_left_plies, m.moves_left_alpha);
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
    initPlot([], [], [], [], [], []);
    setEval(0, 0.5);
    $('depth').textContent = '0';
    $('sims').textContent = '0';
    $('nps').textContent = '0';
    $('ponderSims').textContent = '—';
    $('ponderNps').textContent = '—';
    $('lastMoveTime').textContent = '—';
    setMovesLeft(null);
    renderCandidates([]);
    renderPV([]);
    lastTop = []; lastOpp = false; drawArrows([]);
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
  setMovesLeft(s.moves_left_plies, s.moves_left_alpha);
  const hasMcts = s.win_prob_series && s.win_prob_series.length > 0;
  const hasNn = s.nn_win_prob_series && s.nn_win_prob_series.length > 0;
  if (hasMcts || hasNn) {
    const xs  = hasMcts ? s.win_prob_series.map(p => p[0]) : [];
    const ys  = hasMcts ? s.win_prob_series.map(p => p[1]) : [];
    const nxs = hasNn   ? s.nn_win_prob_series.map(p => p[0]) : [];
    const nys = hasNn   ? s.nn_win_prob_series.map(p => p[1]) : [];
    // Build band arrays from nn_std_series (keyed by ply). Missing std -> zero
    // band radius, so the fill region collapses to the center line for that pt.
    const stdMap = {};
    (s.nn_std_series || []).forEach(p => { stdMap[p[0]] = p[1]; });
    const nLo = nxs.map((x, i) => clamp01(nys[i] - (stdMap[x] || 0)));
    const nHi = nxs.map((x, i) => clamp01(nys[i] + (stdMap[x] || 0)));
    initPlot(xs, ys, nxs, nys, nLo, nHi);
    if (hasMcts) setEval(null, ys[ys.length - 1]);
  }
  if (s.last_tick) applyTick(s.last_tick);
  if (s.last_ponder_tick && s.last_ponder_tick.status !== 'stopped') {
    $('ponderSims').textContent = s.last_ponder_tick.sims.toLocaleString();
    $('ponderNps').textContent = s.last_ponder_tick.nps.toLocaleString();
    // Restore predicted-opponent arrows if we refreshed mid-ponder.
    if (s.last_ponder_tick.top) {
      lastTop = s.last_ponder_tick.top; lastOpp = true; drawArrows(lastTop, true);
    }
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
    case 'tick':     applyTick(ev); lastTop = ev.top || []; lastOpp = false;
                     drawArrows(lastTop, false); break;
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
        // Predicted opponent replies, drawn in the opponent (blue) colour.
        if (ev.top) { lastTop = ev.top; lastOpp = true; drawArrows(lastTop, true); }
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
