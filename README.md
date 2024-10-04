# DQNs

Implementation of DQNs.

Environment : 
[OpenAI Gym Atari 2600 games](https://gym.openai.com/envs/#atari)

## Papers
DQN : [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

Double DQN : [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)

Prioritized Replay : [PRIORITIZED EXPERIENCE REPLAY](https://arxiv.org/pdf/1511.05952.pdf)

Dueling Network : [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)

Ape-X DQN : [DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://openreview.net/pdf?id=H1Dy---0Z)

## Usage

```
$ python dqn_atari.py --prioritized --double --dueling --n_step 3
```
`--prioritezed`: Prioritized Experience Replay

`--double`: Double Deep Q Learning (DDQN)

`--dueling`: Dueling Network

`--n_step <int>`: Multi-step bootstrap target


Other arguments are described in `dqn_atari.py`


### Ape-X DQN
See  https://github.com/omurammm/apex_dqn


## Results
After 12,000 episodes (Ape-X DQN)

![apex](https://user-images.githubusercontent.com/39490801/42048593-abbf5fa0-7b3e-11e8-9301-8690b24edc50.gif)


### Learning curves
<img width="408" alt="2018-06-20 14 38 05" src="https://user-images.githubusercontent.com/39490801/41701914-33c704a6-7569-11e8-9952-6f1884965b57.png">
