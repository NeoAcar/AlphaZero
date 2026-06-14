"""Compare a REUSED grandchild subtree against a FRESH search of the same sim
count from that grandchild's position, to test whether pruning preserves the
subtree (i.e. whether tree reuse is "free").

    1. Search 100 sims at position P.
    2. Take the most-visited child -> its most-visited grandchild (visited N times).
    3. Fresh-search N sims from that grandchild's board.
    4. Print both subtrees' child stats and render compare_reuse.png.

Uses the real deployed 19-plane model. Sequential MCTS (deterministic, eps=0) so
any difference is structural, not noise.

    uv run python scripts/compare_reuse.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import chess
import torch

from alphazero import utils as f
from alphazero.mcts import MCTS
from alphazero.nn import SEResNet, SEResNetWDL, detect_in_channels

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "models/model_best_combined_wdl.pth"
FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
SIMS = 100
MAXD, TOPK = 3, 4

st = torch.load(CKPT, map_location=DEV, weights_only=False)
in_ch = detect_in_channels(st)
model = (SEResNetWDL if "wdl" in CKPT else SEResNet)(in_channels=in_ch).to(DEV)
model.load_state_dict(st["model_state_dict"])
model.eval()
print(f"model in_channels={in_ch}, device={DEV}")

ARGS = dict(device=DEV, c_base=19652.0, c_init=1.33, c_fpu=0.2, t=1.0,
            action_space=4672, dirichlet_epsilon=0.0, dirichlet_alpha=0.3,
            input_planes=in_ch, value_scalar="expected")

board = chess.Board(FEN)

# 1) 100-sim search at P, take most-visited child -> most-visited grandchild.
m1 = MCTS(dict(ARGS, num_simulation=SIMS), model)
m1.set_rep_counter({board._transposition_key(): 1})
m1.search(board.copy(), 0)
child = max(m1.root.children.values(), key=lambda c: c.N)
gchild = max(child.children.values(), key=lambda c: c.N)
Ng = gchild.N
print(f"grandchild visited {Ng} times during the {SIMS}-sim search")

# 2) fresh search from the grandchild's position with Ng sims.
m2 = MCTS(dict(ARGS, num_simulation=Ng), model)
m2.set_rep_counter({gchild.state._transposition_key(): 1})
m2.search(gchild.state.copy(), gchild.move_counter)
fresh = m2.root


def kids(node):
    return sorted(node.children.items(), key=lambda kv: -kv[1].N)


def show(tag, node):
    rows = [(f.alphazero_to_move(a, node.state), c.N, round(c.Q / max(c.N, 1), 3))
            for a, c in kids(node)]
    print(f"{tag}: N={node.N}, children={len(node.children)} -> {rows}")


show("REUSED grandchild", gchild)
show("FRESH  same pos  ", fresh)


# 3) draw both trees.
def layout(node):
    pos, counter = {}, [0]

    def rec(n, d):
        if d >= MAXD or not n.children:
            pos[id(n)] = (counter[0], -d); counter[0] += 1
            return counter[0] - 1
        xs = [rec(c, d + 1) for _, c in kids(n)[:TOPK]]
        pos[id(n)] = (sum(xs) / len(xs), -d)
        return pos[id(n)][0]

    rec(node, 0)
    return pos


def draw(ax, node, title):
    pos = layout(node)

    def rec(n, d):
        x, y = pos[id(n)]
        if d < MAXD:
            for _, c in kids(n)[:TOPK]:
                if id(c) in pos:
                    cx, cy = pos[id(c)]
                    ax.plot([x, cx], [y, cy], "-", color="#888", lw=0.8, zorder=1)
                    rec(c, d + 1)
        q = n.Q / max(n.N, 1)
        ax.scatter([x], [y], s=420, color=plt.cm.RdYlGn((q + 1) / 2),
                   edgecolors="k", linewidths=0.5, zorder=2)
        ax.text(x, y - 0.22, f"N={n.N}\nQ={q:+.2f}", ha="center", va="top", fontsize=6)

    rec(node, 0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(16, 8))
draw(axes[0], gchild, f"REUSED grandchild subtree (N={gchild.N})")
draw(axes[1], fresh, f"FRESH from same position ({fresh.N} sims)")
fig.suptitle("Reused subtree vs fresh search — same position, same sim count", fontsize=13)
fig.tight_layout()
fig.savefig("compare_reuse.png", dpi=120)
print("saved compare_reuse.png")
