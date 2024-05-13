# GPU accelerated Counterfactuar Regret Minimization
Source code for thesis project "GPU accelerated Counterfactual Regret Minimization" at KTH by Gustaf Bergmark.
The repository contains two branches:
### fixed-flop
This branch contains the source code used for measuring the efficiency of GPU and CPU implementations on a fixed flop subgame.

### full-game
This branch contains the source code used for evalutaing the hybrid recall abstraction and scalability of the GPU version.

## Requirements
A GPU capable of running CUDA 12.2, and 64 GB of RAM if running `full-game`

## Quickstart

To run the CPU implementation, use:

`cargo run --release`

To run the GPU implementation, use: 

`cargo run --release --features GPU`
