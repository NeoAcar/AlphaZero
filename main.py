import torch 
import numpy as np
import time
import optimized_functions as f
from mcts import MCTS # ithinkbettermcts.py can be used here
from resnet import ResNet
import chess



class AlphaZero:
    def __init__(self, args):
        self.args = args
        self.model = ResNet().to(args['device'])
        self.mcts = MCTS(args, self.model)

    def game_policy(self, state: chess.Board, move_counter: int) -> tuple[str, np.ndarray]:
        """
        Determine the best move for a given board state using MCTS.
        """
        mcts_action_probs = self.mcts.search(state, move_counter)
        action = np.argmax(mcts_action_probs)
        return f.alphazero_to_move(action), mcts_action_probs
    @torch.no_grad()
    def self_play(self) -> list[chess.Board]:
        """
        Simulate a single self-play game, returning the list of board states.
        """
        
        states = []
        state = chess.Board()
        board = chess.Board()

        move_counter = 0
        while not f.game_result(state, move_counter, self.args["truncation"])[1]:
            print(f"Move {move_counter + 1}")
            action, _ = self.game_policy(state, move_counter)
            state.push_san(action)
            states.append(state.copy())
            state = state.mirror()
            move_counter += 1
            if move_counter % 2 == 1:
                board.push_san(action)
            else:
                board.push_san(f.mirror_move(action))
        print(f"Game result: {f.game_result(state, move_counter, self.args['truncation'])[0]}")
        return states, board
   

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = {
        "num_simulation":400,# paperde 800 bu .d :):):):):) .d 💀
        "truncation": 50,
        "c_base": 19652,
        "c_init": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.03,
        "memory_size": 1000,
        "action_space": 4672,
        "device": device,
        "top_actions": 5,
        "t": 1,
        "model_checkpoint_path": "model_epoch_1.pth",
    }
    A0 = AlphaZero(args)
    checkpoint_path = args["model_checkpoint_path"]
    A0.model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    t = time.time()
    results = A0.self_play()

    print(f"Elapsed time: {time.time() - t}")
    print(f"Finished self-play.")
    print(f"Board fen: {results[1].fen()}")
 
    print(chess.Board().variation_san(results[1].move_stack))
  

    
