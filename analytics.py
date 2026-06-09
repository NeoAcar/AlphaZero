"""Analytics ingest + dashboard.

Walks `data/selfplay/` (one subdir per checkpoint version) and
`logs/matches/` (ladder JSONs) and produces:

  * a CSV summary at <output>/analytics.csv
  * a self-contained HTML dashboard at <output>/dashboard.html (plotly)

Per checkpoint we report:
  - games generated to date
  - avg game length
  - decisive / draw / truncation split
  - PCR ratio (high-sim positions / total)
  - resign rate
  - relative + cumulative Elo from the ladder runs

Usage:
    uv run python analytics.py \\
        --selfplay-dir data/selfplay \\
        --ladder-dir logs/matches \\
        --output logs/analytics
"""
import argparse
import csv
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn


def load_selfplay_summaries(root: Path) -> dict:
    """Returns {checkpoint_name: aggregated_stats_dict}."""
    by_ckpt: dict = {}
    if not root.exists():
        return by_ckpt
    for gen_dir in sorted(root.iterdir(), key=lambda d: d.stat().st_mtime):
        if not gen_dir.is_dir():
            continue
        ck = gen_dir.name
        agg = by_ckpt.setdefault(ck, {
            "checkpoint":     ck,
            "sessions":       0,
            "games":          0,
            "positions":      0,
            "wins_white":     0,
            "wins_black":     0,
            "draws":          0,
            "truncated":      0,
            "resigned":       0,
            "checkmated":     0,
            "rule_draws":     0,
            "high_sim_positions": 0,
            "low_sim_positions":  0,
            "duration_s":     0.0,
            "first_session":  None,
            "last_session":   None,
        })
        for jf in sorted(gen_dir.glob("games_*.json")):
            try:
                m = json.loads(jf.read_text())
            except Exception:
                continue
            agg["sessions"]   += 1
            agg["games"]      += int(m.get("games_completed", 0))
            agg["positions"]  += int(m.get("positions_total", m.get("positions", 0)))
            agg["wins_white"] += int(m.get("wins_white", 0))
            agg["wins_black"] += int(m.get("wins_black", 0))
            agg["draws"]      += int(m.get("draws", 0))
            agg["truncated"]  += int(m.get("truncated", 0))
            agg["resigned"]   += int(m.get("resigned", 0))
            agg["checkmated"] += int(m.get("checkmated", 0))
            agg["rule_draws"] += int(m.get("rule_draws", 0))
            agg["high_sim_positions"] += int(m.get("high_sim_positions", 0))
            agg["low_sim_positions"]  += int(m.get("low_sim_positions", 0))
            agg["duration_s"] += float(m.get("duration_s", 0.0))
            t = m.get("started")
            if t:
                if agg["first_session"] is None or t < agg["first_session"]:
                    agg["first_session"] = t
                if agg["last_session"] is None or t > agg["last_session"]:
                    agg["last_session"] = t
    # Derived fields.
    for ck, a in by_ckpt.items():
        a["avg_plies"]     = (a["positions"] / a["games"]) if a["games"] else 0.0
        a["decisive_rate"] = ((a["checkmated"] + a["resigned"]) / a["games"]) if a["games"] else 0.0
        a["draw_rate"]     = (a["draws"] / a["games"]) if a["games"] else 0.0
        a["trunc_rate"]    = (a["truncated"] / a["games"]) if a["games"] else 0.0
        total_pcr = a["high_sim_positions"] + a["low_sim_positions"]
        a["pcr_high_frac"] = (a["high_sim_positions"] / total_pcr) if total_pcr else 0.0
    return by_ckpt


def load_opening_lines(selfplay_root: Path, max_plies: int = 6,
                       top_n: int = 10) -> dict:
    """For each checkpoint subdirectory, parse all .pgn files. Compute:
      - top-N most-played opening sequences (first `max_plies` SAN moves)
      - exact full-game duplicate count (hash of complete UCI sequence)

    Returns per checkpoint:
        "games":           int                       # total parsed
        "top":             list[(opening_tuple, count, percent)]
        "unique":          int                       # distinct opening tuples
        "convergence":     float                     # share of games in top_n
        "unique_games":    int                       # distinct full sequences
        "dup_max":         int                       # max duplicate count
        "dup_top":         list[(short_uci_str, count)]  # top duplicates
    """
    out: dict = {}
    if not selfplay_root.exists():
        return out
    for gen_dir in sorted(selfplay_root.iterdir(), key=lambda d: d.stat().st_mtime):
        if not gen_dir.is_dir():
            continue
        opening_counter: Counter = Counter()
        full_counter: Counter = Counter()
        full_preview: dict = {}            # hash → first-N-moves SAN preview
        n_games = 0
        for pgn_path in sorted(gen_dir.glob("games_*.pgn")):
            text = pgn_path.read_text()
            stream = io.StringIO(text)
            while True:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                n_games += 1
                board = game.board()
                moves_san: list = []
                moves_uci: list = []
                for mv in game.mainline_moves():
                    moves_san.append(board.san(mv))
                    moves_uci.append(mv.uci())
                    board.push(mv)
                if not moves_uci:
                    continue
                # Opening tuple (first max_plies SAN).
                if moves_san[:max_plies]:
                    opening_counter[tuple(moves_san[:max_plies])] += 1
                # Full-game hash. Use UCI tuple as the key.
                full_key = tuple(moves_uci)
                full_counter[full_key] += 1
                if full_key not in full_preview:
                    # Keep an 8-ply SAN preview for the dashboard.
                    full_preview[full_key] = " ".join(moves_san[:8]) + (
                        " …" if len(moves_san) > 8 else "")
        if n_games == 0:
            continue
        top = opening_counter.most_common(top_n)
        top_pct = [(seq, n, n * 100.0 / n_games) for seq, n in top]
        in_top = sum(n for _, n in top)
        # Full-game dup stats.
        dup_top_raw = [(k, v) for k, v in full_counter.most_common(5) if v > 1]
        dup_top = [(full_preview[k], v) for k, v in dup_top_raw]
        out[gen_dir.name] = {
            "games":        n_games,
            "top":          top_pct,
            "unique":       len(opening_counter),
            "convergence":  (in_top / n_games) if n_games else 0.0,
            "unique_games": len(full_counter),
            "dup_max":      max((v for _, v in full_counter.most_common(1)), default=1),
            "dup_top":      dup_top,
        }
    return out


def load_ladder_results(root: Path) -> list:
    """Returns a list of ladder dicts (one per ladder JSON)."""
    out = []
    if not root.exists():
        return out
    for jf in sorted(root.glob("*_ladder.json"), key=lambda p: p.stat().st_mtime):
        try:
            out.append(json.loads(jf.read_text()))
        except Exception:
            continue
    return out


def write_csv(by_ckpt: dict, ladder: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Compute cumulative Elo: for each ladder, find the "seed" opponent's
    # Elo (anchor at 0) and the new model's Elo as anchor + diff.
    # Simplification: each ladder reports overall_elo_diff vs *the average* of
    # opponents. We'll use that as the per-ladder gain. Cumulative = sum.
    cum_elo = 0.0
    ck_to_ladder = {}
    for l in ladder:
        ck = l.get("new_name") or Path(l.get("new", "")).stem
        gain = l.get("overall_elo_diff") or 0.0
        cum_elo += gain
        ck_to_ladder[ck] = {
            "elo_gain_vs_field": gain,
            "cum_elo":           cum_elo,
            "ladder_games":      l.get("total_wins", 0) + l.get("total_draws", 0) + l.get("total_losses", 0),
        }
    for ck in sorted(by_ckpt):
        a = by_ckpt[ck]
        ld = ck_to_ladder.get(ck, {})
        rows.append({
            "checkpoint":           ck,
            "first_session":        a.get("first_session"),
            "last_session":         a.get("last_session"),
            "sessions":             a["sessions"],
            "games":                a["games"],
            "positions":            a["positions"],
            "avg_plies":            round(a["avg_plies"], 1),
            "decisive_rate":        round(a["decisive_rate"], 3),
            "draw_rate":            round(a["draw_rate"], 3),
            "trunc_rate":           round(a["trunc_rate"], 3),
            "pcr_high_frac":        round(a["pcr_high_frac"], 3),
            "wins_white":           a["wins_white"],
            "wins_black":           a["wins_black"],
            "elo_gain_vs_field":    ld.get("elo_gain_vs_field"),
            "cum_elo":              ld.get("cum_elo"),
            "ladder_games":         ld.get("ladder_games"),
            "duration_s":           round(a["duration_s"], 1),
        })
    if not rows:
        out_path.write_text("(no data)\n")
        return
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _format_opening_html(openings: dict) -> str:
    """Render the per-checkpoint top-N opening lines + full-game duplicate
    section as a series of cards (one per checkpoint generation)."""
    if not openings:
        return "<p style='color:#888'>No opening data yet.</p>"
    cards = []
    for ck, data in openings.items():
        rows = []
        for seq, n, pct in data["top"]:
            san = " ".join(seq)
            rows.append(
                f"<tr><td>{n}</td><td>{pct:.1f}%</td>"
                f"<td style='font-family:ui-monospace,monospace'>{san}</td></tr>"
            )
        rows_html = "".join(rows) if rows else "<tr><td colspan=3>—</td></tr>"
        # Full-game duplicate block.
        n_games = data["games"]
        unique_games = data["unique_games"]
        dup_max = data["dup_max"]
        dup_html = ""
        if dup_max > 1:
            dup_rows = "".join(
                f"<tr><td>{count}</td><td style='font-family:ui-monospace,"
                f"monospace;font-size:11px'>{prev}</td></tr>"
                for prev, count in data["dup_top"]
            )
            dup_html = (
                f"<div class='dup-block'>"
                f"<div class='dup-head'>identical games: "
                f"<b>{n_games - unique_games}</b> duplicates "
                f"({unique_games}/{n_games} unique, max ×{dup_max})</div>"
                f"<table class='opening-table'>"
                f"<thead><tr><th>×</th><th>preview</th></tr></thead>"
                f"<tbody>{dup_rows}</tbody></table></div>"
            )
        else:
            dup_html = (
                f"<div class='dup-block dup-clean'>"
                f"identical games: <b>0</b> "
                f"({unique_games}/{n_games} unique)</div>"
            )
        cards.append(f"""
<div class="opening-card">
  <div class="opening-head">
    <span class="opening-ck">{ck}</span>
    <span class="opening-meta">{n_games} games  ·
      {data['unique']} unique opening lines  ·
      top-{len(data['top'])} cover {data['convergence']*100:.0f}%</span>
  </div>
  <table class="opening-table">
    <thead><tr><th>n</th><th>%</th><th>line</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  {dup_html}
</div>
""")
    return "<div class='opening-grid'>" + "".join(cards) + "</div>"


def render_html(by_ckpt: dict, ladder: list, out_path: Path,
                openings: dict | None = None) -> None:
    """Plotly HTML dashboard. plotly is bundled as a CDN script -- no extra
    Python dep needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ck_list = sorted(by_ckpt.keys())
    openings = openings or {}

    # Per-checkpoint series.
    plies      = [by_ckpt[ck]["avg_plies"]     for ck in ck_list]
    decisive   = [by_ckpt[ck]["decisive_rate"] for ck in ck_list]
    draw_rate  = [by_ckpt[ck]["draw_rate"]     for ck in ck_list]
    trunc_rate = [by_ckpt[ck]["trunc_rate"]    for ck in ck_list]
    games_cum  = []
    running = 0
    for ck in ck_list:
        running += by_ckpt[ck]["games"]
        games_cum.append(running)

    # Ladder Elo series, indexed by checkpoint where possible.
    ladder_ck    = []
    ladder_elo   = []
    cum = 0.0
    for l in ladder:
        ck = l.get("new_name") or Path(l.get("new", "")).stem
        gain = l.get("overall_elo_diff") or 0.0
        cum += gain
        ladder_ck.append(ck)
        ladder_elo.append(cum)

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>AlphaZero training analytics</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font: 14px/1.4 system-ui, sans-serif; background: #161616; color: #ddd;
         margin: 0; padding: 20px; }}
  h1 {{ font-size: 18px; color: #fff; margin: 0 0 20px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .card {{ background: #1f1f1f; border: 1px solid #2a2a2a;
           border-radius: 8px; padding: 12px; }}
  .full {{ grid-column: 1 / -1; }}
  .opening-grid {{ display: grid;
                   grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
                   gap: 16px; margin-top: 16px; }}
  .opening-card {{ background: #181818; border: 1px solid #2a2a2a;
                   border-radius: 6px; padding: 10px 14px; }}
  .opening-head {{ display: flex; justify-content: space-between;
                   align-items: baseline; margin-bottom: 6px; }}
  .opening-ck {{ font-weight: 600; color: #fff; }}
  .opening-meta {{ color: #888; font-size: 11px; }}
  .opening-table {{ width: 100%; font-size: 12px; border-collapse: collapse; }}
  .opening-table th {{ text-align: left; color: #888; font-weight: 400;
                       border-bottom: 1px solid #2a2a2a; padding: 4px 6px 4px 0; }}
  .opening-table td {{ padding: 3px 6px 3px 0; vertical-align: top; }}
  .opening-table td:nth-child(1) {{ color: #66bb6a; width: 30px; }}
  .opening-table td:nth-child(2) {{ color: #aaa; width: 50px; }}
  .dup-block {{ margin-top: 10px; padding-top: 8px;
                border-top: 1px solid #2a2a2a; font-size: 12px; }}
  .dup-head {{ color: #ffa726; margin-bottom: 4px; }}
  .dup-clean {{ color: #66bb6a; }}
  h2 {{ font-size: 14px; margin: 24px 0 8px; color: #ccc;
        text-transform: uppercase; letter-spacing: 0.5px; }}
</style></head><body>
<h1>AlphaZero training analytics ({datetime.now():%Y-%m-%d %H:%M})</h1>
<div class="grid">
  <div class="card full"><div id="elo"></div></div>
  <div class="card"><div id="games"></div></div>
  <div class="card"><div id="plies"></div></div>
  <div class="card"><div id="outcome"></div></div>
  <div class="card"><div id="pcr"></div></div>
  <div class="card full"><div id="conv"></div></div>
</div>
<h2>Opening lines per checkpoint</h2>
{_format_opening_html(openings)}
<script>
const ck = {json.dumps(ck_list)};
const elo_ck = {json.dumps(ladder_ck)};
const elo_y  = {json.dumps(ladder_elo)};
const games_cum = {json.dumps(games_cum)};
const plies     = {json.dumps(plies)};
const decisive  = {json.dumps(decisive)};
const draw_rate = {json.dumps(draw_rate)};
const trunc     = {json.dumps(trunc_rate)};
const pcr       = {json.dumps([by_ckpt[c]["pcr_high_frac"] for c in ck_list])};

const dark = {{
  paper_bgcolor: '#1f1f1f', plot_bgcolor: '#1f1f1f',
  font: {{ color: '#bbb', size: 11 }},
  xaxis: {{ gridcolor: '#2a2a2a' }}, yaxis: {{ gridcolor: '#2a2a2a' }},
  margin: {{ l: 50, r: 20, t: 30, b: 40 }}
}};

Plotly.newPlot('elo', [
  {{ x: elo_ck, y: elo_y, mode: 'lines+markers',
     name: 'cum Elo (vs ladder field)',
     line: {{ color: '#66bb6a', width: 2.5 }},
     marker: {{ size: 8, color: '#66bb6a' }} }}
], {{ ...dark, title: 'Cumulative Elo over checkpoints',
      xaxis: {{ title: 'checkpoint', ...dark.xaxis }},
      yaxis: {{ title: 'Elo gain (cumulative)', ...dark.yaxis }} }},
   {{ displayModeBar: false }});

Plotly.newPlot('games', [
  {{ x: ck, y: games_cum, mode: 'lines+markers', name: 'cumulative games',
     line: {{ color: '#42a5f5' }}, marker: {{ color: '#42a5f5' }} }}
], {{ ...dark, title: 'Self-play games generated (cumulative)',
      xaxis: {{ ...dark.xaxis }}, yaxis: {{ ...dark.yaxis }} }},
   {{ displayModeBar: false }});

Plotly.newPlot('plies', [
  {{ x: ck, y: plies, mode: 'lines+markers', name: 'avg plies/game',
     line: {{ color: '#ffa726' }}, marker: {{ color: '#ffa726' }} }}
], {{ ...dark, title: 'Average game length',
      yaxis: {{ title: 'plies', ...dark.yaxis }},
      xaxis: {{ ...dark.xaxis }} }},
   {{ displayModeBar: false }});

Plotly.newPlot('outcome', [
  {{ x: ck, y: decisive, name: 'decisive', type: 'bar',
     marker: {{ color: '#66bb6a' }} }},
  {{ x: ck, y: draw_rate, name: 'draw',     type: 'bar',
     marker: {{ color: '#9e9e9e' }} }},
  {{ x: ck, y: trunc,     name: 'truncated', type: 'bar',
     marker: {{ color: '#ef5350' }} }},
], {{ ...dark, title: 'Outcome split per checkpoint',
      barmode: 'stack',
      yaxis: {{ range: [0, 1], tickformat: '.0%', ...dark.yaxis }},
      xaxis: {{ ...dark.xaxis }} }},
   {{ displayModeBar: false }});

Plotly.newPlot('pcr', [
  {{ x: ck, y: pcr, mode: 'lines+markers', name: 'high-sim fraction',
     line: {{ color: '#ab47bc' }}, marker: {{ color: '#ab47bc' }} }}
], {{ ...dark, title: 'PCR high-sim fraction (should track p_high config)',
      yaxis: {{ range: [0, 1], tickformat: '.0%', ...dark.yaxis }},
      xaxis: {{ ...dark.xaxis }} }},
   {{ displayModeBar: false }});

const conv_ck = {json.dumps(list(openings.keys()))};
const conv_y  = {json.dumps([round(openings[c]["convergence"] * 100, 1) for c in openings])};
const unique  = {json.dumps([openings[c]["unique"] for c in openings])};
const games_oc = {json.dumps([openings[c]["games"] for c in openings])};
Plotly.newPlot('conv', [
  {{ x: conv_ck, y: conv_y, mode: 'lines+markers',
     name: 'top-10 coverage %',
     line: {{ color: '#ffca28', width: 2.5 }},
     marker: {{ size: 8, color: '#ffca28' }} }},
  {{ x: conv_ck, y: unique, mode: 'lines+markers',
     name: 'unique opening lines',
     yaxis: 'y2',
     line: {{ color: '#7e57c2', width: 1.5, dash: 'dot' }},
     marker: {{ size: 5, color: '#7e57c2' }} }},
], {{ ...dark,
      title: 'Opening convergence (higher = more concentrated mainlines)',
      xaxis: {{ ...dark.xaxis }},
      yaxis: {{ title: 'top-10 coverage %', range: [0, 100], ...dark.yaxis }},
      yaxis2: {{ title: 'unique lines', overlaying: 'y', side: 'right',
                gridcolor: '#2a2a2a' }} }},
   {{ displayModeBar: false }});
</script>
</body></html>"""
    out_path.write_text(html)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selfplay-dir", default="data/selfplay")
    p.add_argument("--ladder-dir",   default="logs/matches")
    p.add_argument("--output",       default="logs/analytics")
    p.add_argument("--opening-plies", type=int, default=6,
                   help="how many SAN plies define an 'opening' line (default 6 = 3 full moves)")
    p.add_argument("--opening-top",   type=int, default=10,
                   help="how many top lines to keep per checkpoint")
    cli = p.parse_args()

    out_dir = Path(cli.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ingesting self-play summaries from {cli.selfplay_dir}")
    by_ckpt = load_selfplay_summaries(Path(cli.selfplay_dir))
    print(f"  {len(by_ckpt)} checkpoint generation(s) seen")
    for ck, a in by_ckpt.items():
        print(f"    {ck}: {a['games']} games, {a['positions']} positions")

    print(f"Ingesting ladder results from {cli.ladder_dir}")
    ladder = load_ladder_results(Path(cli.ladder_dir))
    print(f"  {len(ladder)} ladder run(s) found")

    print(f"Parsing PGNs for opening lines (first {cli.opening_plies} plies)")
    openings = load_opening_lines(
        Path(cli.selfplay_dir), max_plies=cli.opening_plies, top_n=cli.opening_top,
    )
    for ck, d in openings.items():
        print(f"  {ck}: {d['games']} games, {d['unique']} unique lines, "
              f"top-{len(d['top'])} covers {d['convergence']*100:.0f}%")

    csv_path  = out_dir / "analytics.csv"
    html_path = out_dir / "dashboard.html"
    write_csv(by_ckpt, ladder, csv_path)
    render_html(by_ckpt, ladder, html_path, openings=openings)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {html_path}")
    print(f"Open file://{html_path.resolve()} to view.")


if __name__ == "__main__":
    main()
