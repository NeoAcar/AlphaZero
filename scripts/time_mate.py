"""Time how long a proven-mate move actually takes in pure search (warm),
to separate search cost from uci.py per-move overhead."""
import os, sys, time
os.environ["UCI_MONITOR_URL"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess, torch
from alphazero import utils as f
from alphazero.batched_mcts import BatchedMCTS
from alphazero.nn import SEResNetWDL, detect_in_channels

DEV = "cuda" if torch.cuda.is_available() else "cpu"
st = torch.load("models/model_best_combined_wdl.pth", map_location=DEV, weights_only=False)
in_ch = detect_in_channels(st)
model = SEResNetWDL(in_channels=in_ch).to(DEV)
if DEV == "cuda": model = model.to(memory_format=torch.channels_last)
model.load_state_dict(st["model_state_dict"]); model.eval()

ARGS = dict(num_simulation=1200, truncation=1000, c_base=19652.0, c_init=1.33,
            c_fpu=0.2, t=1.0, action_space=4672, dirichlet_epsilon=0.0,
            dirichlet_alpha=0.3, input_planes=in_ch, value_scalar="expected",
            batch_size=8, early_stop=False, tree_reuse=False, device=DEV,
            mate_stop=True)

MATE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"   # Ra8#

def timed(fen, label, n=3):
    board = chess.Board(fen)
    for i in range(n):
        m = BatchedMCTS(dict(ARGS), model)
        m.set_rep_counter({board._transposition_key(): 1})
        t0 = time.monotonic()
        m.search(board.copy(), 0)
        dt = time.monotonic() - t0
        tag = "warmup" if i == 0 else f"run{i}"
        print(f"{label:20} {tag}: {dt*1000:7.1f} ms   root.N={m.root.N}")

timed(MATE, "mate (Ra8#)")
timed(chess.STARTING_FEN, "startpos 1200")
