# rl_implementation

Implementation of papers about Reinforcement Learning.

Environment : 
[OpenAI Gym Atari 2600 games](https://gym.openai.com/envs/#atari)

## Ape-X DQN
https://openreview.net/pdf?id=H1Dy---0Z

See  https://github.com/utarumo/apex_dqn

## Double Deep Q Learning (DDQN)
https://arxiv.org/pdf/1509.06461.pdf

`ddqn_atari.py`

## Dueling Network
https://arxiv.org/pdf/1511.06581.pdf

`dueling_ddqn_atari.py`

## Prioritized Experience Replay
https://arxiv.org/pdf/1511.05952.pdf

`ddqn_per_atari.py`
`ddqn_is_per_atari.py`
(with Importance Sampling)

# Result
After 12,000 episodes (Ape-X DQN)

![apex](https://user-images.githubusercontent.com/39490801/42048593-abbf5fa0-7b3e-11e8-9301-8690b24edc50.gif)

<img width="408" alt="2018-06-20 14 38 05" src="https://user-images.githubusercontent.com/39490801/41701914-33c704a6-7569-11e8-9952-6f1884965b57.png">
