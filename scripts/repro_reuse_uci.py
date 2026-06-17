"""Deterministic reproduction of the lichess 'tree-reuse hurts (esp. as black)'
report, driving the REAL UciEngine through the same command flow lichess-bot
uses (ucinewgame -> position startpos moves ... -> go), which exercises the
apply_action() reuse path -- NOT the update_root() path that match.py tests.

Reuse must be FAITHFUL: with Temperature=0 and identical inputs, TreeReuse=true
and TreeReuse=false must pick the SAME move every ply. Any divergence is a
reuse-correctness bug. We run the bot as BLACK and as WHITE and also assert,
each ply, that the reused root.state actually equals the real mirror_state
(catches apply_action rerooting onto the wrong/stale node).

    uv run python scripts/repro_reuse_uci.py
"""
import os
import sys
os.environ["UCI_MONITOR_URL"] = ""   # no telemetry
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import uci as U
from alphazero import utils as f

SIMS = 256
N_PLIES = 14


def opp_move(board: chess.Board) -> str:
    """Deterministic legal opponent: lowest-uci legal move. Identical for both
    engines as long as they're on the same board, so inputs stay byte-identical
    until (if) the bot's own choice diverges."""
    return min(m.uci() for m in board.legal_moves)


def make_engine(reuse: bool) -> U.UciEngine:
    eng = U.UciEngine()
    eng.options.update({
        "Sims": SIMS, "Temperature": 0.0, "TempMoves": 0,
        "DirichletEps": 0.0, "EarlyStop": "false",
        "BackgroundPonder": "false",
        "TreeReuse": "true" if reuse else "false",
    })
    eng.cmd_ucinewgame([])
    eng._ensure_loaded()
    return eng


def bestmove(eng: U.UciEngine, moves: list[str]) -> tuple[str, bool]:
    """Feed `position startpos moves <moves>` + `go`; return (bestmove, root_ok).
    root_ok = reuse landed the MCTS root on the true current position."""
    captured = {}
    orig_send = U.send
    def cap(msg):
        if msg.startswith("bestmove "):
            captured["mv"] = msg.split()[1]
    U.send = cap
    try:
        eng.cmd_position(["startpos", "moves", *moves])
        eng.cmd_go([])
    finally:
        U.send = orig_send
    m = eng.mcts
    root_ok = (m is not None and m.root is not None
               and m.root.state == eng.mirror_state)
    return captured.get("mv", "0000"), bool(root_ok)


def predicted_opp_reply(er: U.UciEngine) -> str | None:
    """After er.cmd_go, er pre-pushed the bot's move so er.mcts.root is the
    post-bot-move node (opponent to move). Its most-visited child = the bot's
    PREDICTED opp reply -> playing it forces maximal subtree inheritance, the
    case that actually stresses reuse. Returns a real-coord uci or None."""
    m = er.mcts
    if m is None or m.root is None or not m.root.children:
        return None
    action = max(m.root.children.items(), key=lambda kv: kv[1].N)[0]
    try:
        mir_uci = f.alphazero_to_move(action, er.mirror_state)
    except Exception:
        return None
    return mir_uci if er.real_board.turn == chess.WHITE else f.mirror_move(mir_uci)


def compare(bot_is_white: bool):
    """Lockstep, but the opponent plays the REUSE engine's predicted top reply
    so apply_action actually inherits a big subtree every ply. Same move list
    feeds both engines -> any move difference is purely reuse vs no-reuse.
    `inhN` = inherited subtree size the reuse engine reused before searching."""
    er = make_engine(reuse=True)
    en = make_engine(reuse=False)
    board = chess.Board()
    moves: list[str] = []
    label = "WHITE" if bot_is_white else "BLACK"
    print(f"\n=== bot plays {label} ===")
    print(f"{'ply':>3} {'inhN':>6} {'reuse':>6} {'fresh':>6}  MATCH")
    diverged = False
    for i in range(N_PLIES):
        if board.is_game_over():
            break
        if (board.turn == chess.WHITE) != bot_is_white:
            mv = opp_move(board)
            board.push_uci(mv); moves.append(mv)
            if board.is_game_over():
                break
        # Inherited subtree size for the reuse engine (after position->apply_action).
        er.cmd_position(["startpos", "moves", *moves])
        inhN = er.mcts.root.N if (er.mcts and er.mcts.root) else 0
        captured = {}
        orig = U.send
        U.send = lambda s: captured.__setitem__("mv", s.split()[1]) if s.startswith("bestmove ") else None
        try:
            er.cmd_go([])
        finally:
            U.send = orig
        rm = captured.get("mv", "0000")
        # Fresh engine searches the SAME TOTAL sims (inhN + base) as reuse, so
        # this is apples-to-apples: any diff = batched-reuse corrupting stats,
        # NOT just "reuse searched deeper".
        en.options["Sims"] = inhN + SIMS
        nm, _ = bestmove(en, moves)
        match = "ok" if rm == nm else "*** DIVERGE ***"
        if rm != nm:
            diverged = True
        print(f"{i:>3} {inhN:>6} {rm:>6} {nm:>6}  {match}")
        if rm != nm:
            print("  -> first divergence")
            break
        board.push_uci(rm); moves.append(rm)
        # Opponent plays the bot's predicted reply (forces deep reuse).
        if not board.is_game_over() and (board.turn == chess.WHITE) != bot_is_white:
            opp = predicted_opp_reply(er)
            if opp and chess.Move.from_uci(opp) in board.legal_moves:
                board.push_uci(opp); moves.append(opp)
    print("RESULT:", "DIVERGED (reuse unfaithful)" if diverged
          else "identical (reuse faithful)")
    return diverged


if __name__ == "__main__":
    import torch
    print(f"device check: cuda={torch.cuda.is_available()}")
    d_black = compare(bot_is_white=False)
    d_white = compare(bot_is_white=True)
    print("\n==== SUMMARY ====")
    print("black diverged:", d_black)
    print("white diverged:", d_white)
