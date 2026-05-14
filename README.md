# Alpha Zero Implementation

## Description
This project is an implementation based on the AlphaZero paper, featuring a bot that plays games using Monte Carlo Tree Search (MCTS) and a ResNet model. The project includes training, self-play, and necessary helper functions to fully realize the AlphaZero algorithm.

## File Structure

- **`train.py`**: Trains the model from a config file + sharded supervised data.
- **`gen_sf_data.py`**: Builds Stockfish-labelled training shards from a PGN.
- **`selfplay.py`**: Generates self-play games for AlphaZero-style training.
- **`runner.py`**: Orchestrates the self-play → train → match-gate loop.
- **`match.py`**: Plays two players against each other (any combination of NN/MCTS/Stockfish/baselines).
- **`play.py`**: Local terminal UI for human vs bot or bot vs bot.
- **`uci.py`**: UCI protocol wrapper so any chess GUI (or lichess-bot) can drive the engine.
- **`alphazero/`**: Library package.
  - **`mcts.py` / `batched_mcts.py`**: Sequential and batched MCTS.
  - **`resnet.py`**: ResNet body + policy/value heads.
  - **`utils.py`**: Board / move / policy encoding helpers.
  - **`dataset.py`**: PyTorch datasets for supervised shards and self-play `.pt` files.
  - **`players.py`**: Player abstractions (random / piece_value / value_only / policy_only / mcts / stockfish) used by `match.py` and `runner.py`.

See `CLAUDE.md` for conventions (mirror-canonical state, 4672 action space, value perspective, proven-value semantics) and `RECIPE.md` for the comparison vs the AlphaGo Zero / AlphaZero papers.

## Model Weights and Training Data
Due to the large size of the model, Lichess games, and evaluation result files (over 25MB), they are not included in this repository. You can download them from the following link:
[Download Model Weights and Results](https://drive.google.com/drive/folders/12PGjUCOllXaKWY-uzP7fr_iY1kj9HhTz?usp=sharing)

Make sure to download these files before running the project.

## Chess Library Bug Fix
The `chess` library currently has a bug. To fix it, follow these steps:

1. Navigate to `your_env/Lib/chess/__init__.py`.
2. At line 3067, delete the entire `if-else` block.
3. Replace it with a simple `return move`.

This will fix the issue and allow the code to run smoothly.

## Requirements
All dependencies can be found in the `requirements.txt` file.

### CUDA Version Instructions
If you want to use a PyTorch version compatible with CUDA, please check your system's CUDA version and use one of the following commands to install PyTorch:

- **For CUDA 12.1**: `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121`
- **For CUDA 11.8**: `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118`

