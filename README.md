# rl_implementation

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
$ python dqn_atari.py --prioritized 1 --double 1 --dueling 1 --n_step 3
```
### Prioritized Experience Replay
`--prioritezed` : 0 or 1

### Double Deep Q Learning (DDQN)
`--double` : 0 or 1

### Dueling Network
`--dueling` : 0 or 1

### multi-step bootstrap target
`--n_step` : int (1 : normal TD error)
<br>  
Other arguments are described in `dqn_atari.py`


### Ape-X DQN
See  https://github.com/omurammm/apex_dqn


# Result
After 12,000 episodes (Ape-X DQN)

![apex](https://user-images.githubusercontent.com/39490801/42048593-abbf5fa0-7b3e-11e8-9301-8690b24edc50.gif)

<img width="408" alt="2018-06-20 14 38 05" src="https://user-images.githubusercontent.com/39490801/41701914-33c704a6-7569-11e8-9952-6f1884965b57.png">
