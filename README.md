# Neural Monte Carlo Tree Search

An MCTS implementation combined with neural policy/value networks, trained via supervised learning on self-play data.

## Overview

AlphaGo used neural networks with MCTS; this project follows the same approach. It includes:
- MCTS from scratch
- PyTorch models for policy and value
- Data generation via parallel self-play
- Supervised training with augmentation and deduplication

## Technical Highlights

### Monte Carlo Tree Search

Four stages:
- Selection: UCB1
- Expansion: create child states when leaving tree
- Rollout: policy/value from network
- Backpropagation: updates along the path

Custom `NeuralNode` for policy/value-guided search.

### Neural Architecture

Models (`TTTNet` and `UTTTNet`):
- CNNs for board features
- Separate policy and value heads
- Correct handling of canonical states and valid-move masks
- Batchnorm, ReLU, softmax/tanh outputs

### Game Framework

Abstract `Game` interface:
- State, moves, win checks, canonical form, copying
- Implementations for Tic-Tac-Toe and Ultimate Tic-Tac-Toe (81 cells)

### Data Generation and Processing

Steps:
- Multiprocess self-play
- Symmetry augmentation (rotations/flips)
- Deduplication with averaged labels
- Train/val splits

### Memory Management

Weak references to manage large trees and prevent leaks during long runs.

## Requirements

- PyTorch
- NumPy
- Python 3.8+ (multiprocessing, weakref)

## Structure

- `Node`: base MCTS
- `NeuralNode`: MCTS with network guidance
- `Game`: environment interface
- `TTTGame` / `UTTTGame`: game logic
- `TTTNet` / `UTTTNet`: models
- `Arena`: game orchestration
- `Player`: random/MCTS/human/neural agents

## WIP
- Parallel data pipeline (possiblly a distributed pipeline?)
- Migrate jupyter notebook cells to standalone python scripts
  - Scale up the model/training time by running scripts on a cloud provider
