"""AlphaZero chess engine — library code.

Submodules:
    utils         board / move / policy encoding
    nn            ResNet body + policy/value heads
    dataset       PyTorch datasets (supervised shards, self-play .pt)
    mcts          sequential MCTS + Node
    batched_mcts  batched MCTS with virtual loss
    players       player abstractions for match.py / runner.py
"""
