# 📈 Quant-AI: Deep Reinforcement Learning for Algorithmic Trading

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

**Quant-AI** is a research-driven framework designed to explore the application of Deep Reinforcement Learning (DRL) in financial markets. It leverages modern AI architectures to navigate the complexities of market dynamics and optimize trading strategies.

## 🚀 Key Features
- **PPO Agent:** Implementation of Proximal Policy Optimization for stable policy updates.
- **Transformer Encoder:** Utilizes Self-Attention mechanisms to capture long-range temporal dependencies in OHLCV data.
- **Custom Environment:** A robust simulation environment compatible with OpenAI Gym/Gymnasium.
- **Risk Management:** Integrated Reward Shaping focused on Sharpe Ratio and Maximum Drawdown.

## 🛠️ Architecture Overview
`	ext
[ Market Data ] -> [ Transformer Encoder ] -> [ Latent Space ]
                                                     |
                                            -------------------
                                            |                 |
                                      [ Actor (Policy) ] [ Critic (Value) ]
                                            |                 |
                                      [ Buy/Sell/Hold ]   [ Expected Return ]
`

## 📦 Installation
`ash
git clone https://github.com/NavneetPrabhat2/Quant-AI.git
cd Quant-AI
pip install -r requirements.txt
`

## 📊 Quick Start
`python
from quant_ai.agent import TradingAgent
from quant_ai.env import TradingEnv

# Initialize Environment & Agent
env = TradingEnv(data_path="data/nifty50.csv")
agent = TradingAgent(state_dim=env.observation_space.shape, action_dim=3)

# Train the Agent
agent.train(env, episodes=1000)

# Backtest
results = agent.test(env)
print(f"Total Profit: {results['profit']}%")
`

## 📖 Background
This project combines principles from **Quantitative Finance** and **Deep Learning**. By modeling the market as a Markov Decision Process (MDP), we enable agents to learn optimal decision-making policies directly from historical tick data.

---
Developed by **Navneet Prabhat**