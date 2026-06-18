"""Verify the bot stops searching the instant a forced mate FOR US is proven.

With EarlyStop OFF and Sims=1200, a mate-in-1 / mate-in-2 position should
finish in a handful of sims (root.N << 1200), play the mating move, and report
last_was_proven_mate. A non-mate position should run the full budget.

    uv run python scripts/check_mate_stop.py
"""
import os, sys
os.environ["UCI_MONITOR_URL"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess, torch
from alphazero import utils as f
from alphazero.batched_mcts import BatchedMCTS
from alphazero.nn import SEResNetWDL, detect_in_channels

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "models/model_best_combined_wdl.pth"
st = torch.load(CKPT, map_location=DEV, weights_only=False)
in_ch = detect_in_channels(st)
model = SEResNetWDL(in_channels=in_ch).to(DEV)
model.load_state_dict(st["model_state_dict"]); model.eval()

ARGS = dict(num_simulation=1200, truncation=1000, c_base=19652.0, c_init=1.33,
            c_fpu=0.2, t=1.0, action_space=4672, dirichlet_epsilon=0.0,
            dirichlet_alpha=0.3, input_planes=in_ch, value_scalar="expected",
            batch_size=8, early_stop=False, tree_reuse=False, device=DEV)

CASES = [
    ("mate-in-1 (Ra8#)", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),
    ("mate-in-2",         "6k1/5ppp/8/8/8/8/6PP/R5K1 w - - 0 1", None),
    ("no mate (startpos)", chess.STARTING_FEN, None),
]

for label, fen, expect in CASES:
    board = chess.Board(fen)
    m = BatchedMCTS(dict(ARGS), model)
    m.set_rep_counter({board._transposition_key(): 1})
    probs = m.search(board.copy(), 0)
    best = f.alphazero_to_move(int(probs.argmax()), board)
    print(f"\n{label}: root.N={m.root.N}  proven={m.root.proven_value}  "
          f"proven_mate={m.last_was_proven_mate}  best={best}")
    if expect:
        print(f"   expected {expect}: {'OK' if best == expect else 'MISMATCH'}")
    print(f"   {'STOPPED EARLY' if m.root.N < 1200 else 'ran full 1200'}")

# Self-play flag: mate_stop=False must NOT early-stop even on a forced mate.
board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
m = BatchedMCTS(dict(ARGS, mate_stop=False), model)
m.set_rep_counter({board._transposition_key(): 1})
m.search(board.copy(), 0)
print(f"\nself-play (mate_stop=False) on mate-in-1: root.N={m.root.N}  "
      f"{'ran full 1200 (correct)' if m.root.N >= 1200 else 'STOPPED (WRONG)'}")

