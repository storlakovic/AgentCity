# AgentCity

AgentCity is an experimental AI project written in C++. Its long-term goal is to build autonomous agents that live and interact within a simulated city.

Currently, the project focuses on a single agent learning to manage its needs and resources.

## How it works

A custom neural network selects actions based on hunger, energy, money, food, morality, economic conditions, and location. Action masking prevents choices that are unavailable at the current location or with the available resources.

The agent learns through reinforcement learning, collecting experiences across simulated episodes. Rewards encourage survival and account for changes in the agent’s condition.

Training uses batches of episodes and a baseline to stabilize learning. Regular evaluations measure performance in normal and emergency scenarios. Improved models are saved as JSON.

## Implemented features

- Movement between home, workplace, and shop
- Working, buying food, eating, and sleeping
- Food inventory and money management
- Stealing, with effects on morality and possible financial penalties
- An economy factor affecting wages and food prices
- Randomized starting conditions and emergency scenarios
- Metrics for reward, episode length, and survival rate

These mechanics are implemented, and the model has been trained to use them reliably.

## Usage

Choose a mode when starting the program:

- `train`: Train a new model.
- `run`: Load and run a saved model.

## Project status

AgentCity is a personal learning project developed occasionally in spare time. Training, simulation rules, and architecture are gradually being improved.
