import argparse
import sys
import time

import chess
import torch

import optimized_functions as f
from mcts import MCTS
from resnet import ResNet


DEFAULT_ARGS = {
    "num_simulation": 200,
    "truncation": 200,
    "c_base": 19652,
    "c_init": 1.25,
    "dirichlet_epsilon": 0.0,
    "dirichlet_alpha": 0.03,
    "memory_size": 1000,
    "action_space": 4672,
    "top_actions": 5,
    "t": 1,
}


def load_bot(checkpoint_path: str, sims: int) -> tuple[MCTS, dict]:
    args = dict(DEFAULT_ARGS)
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args["num_simulation"] = sims
    model = ResNet().to(args["device"])
    state = torch.load(checkpoint_path, map_location=args["device"], weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return MCTS(args, model), args


def render(board: chess.Board, last_move: chess.Move | None = None) -> None:
    """Print the board from White's perspective with simple unicode pieces."""
    sym = {
        "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
        "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
        ".": "·",
    }
    lines = ["", "    a b c d e f g h", "  +-----------------+"]
    for rank in range(7, -1, -1):
        row = [f"{rank + 1} |"]
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            row.append(sym[piece.symbol()] if piece else sym["."])
        row.append(f"| {rank + 1}")
        lines.append(" ".join(row))
    lines.append("  +-----------------+")
    lines.append("    a b c d e f g h")
    if last_move is not None:
        lines.append(f"\nLast move: {last_move.uci()}")
    lines.append(f"To move: {'White' if board.turn else 'Black'}")
    if board.is_check():
        lines.append("** CHECK **")
    print("\n".join(lines), flush=True)


def bot_move(mcts: MCTS, mirrored_state: chess.Board, move_counter: int) -> str:
    """Run MCTS and return the bot's chosen move as a UCI string in mirrored coords."""
    probs = mcts.search(mirrored_state, move_counter)
    action = int(probs.argmax())
    return f.alphazero_to_move(action, mirrored_state)


def human_move(board: chess.Board) -> chess.Move:
    while True:
        raw = input("Your move (UCI like e2e4, or 'quit'): ").strip()
        if raw.lower() in ("quit", "q", "exit"):
            sys.exit(0)
        try:
            move = chess.Move.from_uci(raw)
        except ValueError:
            try:
                move = board.parse_san(raw)
            except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
                print(f"  invalid: {raw!r}")
                continue
        if move in board.legal_moves:
            return move
        print(f"  illegal in this position: {raw!r}")


def announce_result(real_board: chess.Board, move_counter: int, truncation: int) -> None:
    if real_board.is_checkmate():
        winner = "Black" if real_board.turn else "White"
        print(f"\nCheckmate. {winner} wins.")
    elif real_board.is_stalemate():
        print("\nStalemate. Draw.")
    elif real_board.is_insufficient_material():
        print("\nInsufficient material. Draw.")
    elif real_board.is_fifty_moves():
        print("\nFifty-move rule. Draw.")
    elif move_counter >= truncation:
        print(f"\nTruncated at {move_counter} plies. Draw.")
    else:
        print("\nGame ended.")


def play_human_vs_bot(checkpoint: str, sims: int, human_color: chess.Color, truncation: int) -> None:
    mcts, args = load_bot(checkpoint, sims)
    real_board = chess.Board()
    mirrored_state = chess.Board()
    move_counter = 0

    while not f.game_result(mirrored_state, move_counter, truncation)[1]:
        render(real_board)
        if real_board.turn == human_color:
            mover_was_white = real_board.turn == chess.WHITE
            move = human_move(real_board)
            uci = move.uci()
            real_board.push(move)
            push_uci = uci if mover_was_white else f.mirror_move(uci)
            mirrored_state.push_uci(push_uci)
            mirrored_state = mirrored_state.mirror()
        else:
            print("\nBot is thinking...")
            t0 = time.time()
            uci_mirrored = bot_move(mcts, mirrored_state, move_counter)
            dt = time.time() - t0
            real_uci = uci_mirrored if real_board.turn == chess.WHITE else f.mirror_move(uci_mirrored)
            move = chess.Move.from_uci(real_uci)
            print(f"Bot played {real_uci} ({dt:.1f}s)")
            real_board.push(move)
            mirrored_state.push_uci(uci_mirrored)
            mirrored_state = mirrored_state.mirror()
        move_counter += 1

    render(real_board)
    announce_result(real_board, move_counter, truncation)


def play_bot_vs_bot(checkpoint: str, sims: int, truncation: int, delay: float) -> None:
    mcts, args = load_bot(checkpoint, sims)
    real_board = chess.Board()
    mirrored_state = chess.Board()
    move_counter = 0

    render(real_board)
    while not f.game_result(mirrored_state, move_counter, truncation)[1]:
        t0 = time.time()
        uci_mirrored = bot_move(mcts, mirrored_state, move_counter)
        dt = time.time() - t0
        real_uci = uci_mirrored if real_board.turn == chess.WHITE else f.mirror_move(uci_mirrored)
        move = chess.Move.from_uci(real_uci)
        side = "White" if real_board.turn else "Black"
        print(f"\n[move {move_counter + 1}] {side}: {real_uci}  ({dt:.1f}s)")
        real_board.push(move)
        mirrored_state.push_uci(uci_mirrored)
        mirrored_state = mirrored_state.mirror()
        move_counter += 1
        render(real_board, last_move=move)
        if delay > 0:
            time.sleep(delay)

    announce_result(real_board, move_counter, truncation)
    print("\nPGN of game:")
    print(chess.Board().variation_san(real_board.move_stack))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["human", "bot"], help="human = you vs bot; bot = bot vs bot")
    p.add_argument("--checkpoint", default="model_epoch_1.pth")
    p.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    p.add_argument("--color", choices=["white", "black"], default="white", help="(human mode) which color you play")
    p.add_argument("--truncation", type=int, default=200, help="max plies before draw")
    p.add_argument("--delay", type=float, default=0.0, help="(bot mode) seconds to pause between moves")
    cli = p.parse_args()

    if cli.mode == "human":
        color = chess.WHITE if cli.color == "white" else chess.BLACK
        play_human_vs_bot(cli.checkpoint, cli.sims, color, cli.truncation)
    else:
        play_bot_vs_bot(cli.checkpoint, cli.sims, cli.truncation, cli.delay)


if __name__ == "__main__":
    main()
