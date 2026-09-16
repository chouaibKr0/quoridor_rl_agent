"""
Backward compatibility shim for Quoridor RL Agent training script.
Delegates to solvers.rl.train.
"""

from solvers.rl.train import train, TrainConfig, main

if __name__ == "__main__":
    main()
