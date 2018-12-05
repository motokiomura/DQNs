# coding:utf-8

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input, Lambda, concatenate
from keras import backend as K
import time
from gym import wrappers
import argparse

from SumTree import SumTree


#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.max_p = 1

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def add_p(self, p, sample):
        self.tree.add(p, sample)


    def sample(self, n):
        batch = []
        idx_batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idx_batch.append(idx)

        return batch, idx_batch

    def update(self, idx, error):
        p = self._getPriority(error)
        if p > self.max_p:
            self.max_p = p
        self.tree.update(idx, p)


class Agent():
    def __init__(self,
                 args,
                 num_actions,
                 frame_width = 84,  # Resized frame width
                 frame_height = 84,  # Resized frame height
                 state_length = 4,  # Number of most recent frames to produce the input to the network
                 anealing_steps = 1000000, # Number of steps over which the initial value of epsilon is linearly annealed to its final value
                 initial_epsilon = 1.0,  # Initial value of epsilon in epsilon-greedy
                 final_epsilon = 0.1,  # Final value of epsilon in epsilon-greedy
                 target_update_interval = 10000,  # The frequency with which the target network is updated
                 action_interval = 4,  # The agent sees only every () input
                 train_interval = 4,  # The agent selects 4 actions between successive updates
                 batch_size = 32,  # Mini batch size
                 lr = 0.00025 / 4,  # Learning rate used by RMSProp
                 # MOMENTUM = 0.95  # Momentum used by RMSProp
                 # MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
                 save_interval = 300000,  # The frequency with which the network is saved
                 no_op_steps = 30,  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
                 # initial_beta = 0.4,
                 ):

        self.prioritized = args.prioritized
        self.double = args.double
        self.dueling = args.dueling
        self.n_step = args.n_step

        self.env_name = args.env_name
        self.load = args.load
        self.network_path = args.network_path
        self.initial_replay_size = args.initial_replay_size
        self.gamma = args.gamma
        self.gamma_n = args.gamma ** args.n_step


        self.num_actions = num_actions
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.state_length = state_length
        self.anealing_steps = anealing_steps
        self.target_update_interval = target_update_interval
        self.action_interval = action_interval
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.lr = lr
        self.no_op_steps = no_op_steps
        self.save_interval = save_interval

        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.anealing_by_step = (initial_epsilon - final_epsilon) / anealing_steps
        # self.beta = initial_beta
        # self.beta_step = (1 - initial_epsilon) / args.num_episodes
        self.t = 0
        self.repeated_action = 0




        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        self.start = 0

        # Create replay memory
        #self.replay_memory = deque()

        if self.prioritized:
            self.memory = Memory(args.replay_memory_size)
        else:
            self.memory = deque()

        self.buffer = []
        self.R = 0

        # Dueling Network
        if self.dueling:
            # Create q network
            self.s, self.q_values, q_network = self.build_dueling_network()
            q_network_weights = q_network.trainable_weights

            # Create target network
            self.st, self.target_q_values, target_network = self.build_dueling_network()
            target_network_weights = target_network.trainable_weights

        else:
            # Create q network
            self.s, self.q_values, q_network = self.build_network()
            q_network_weights = q_network.trainable_weights

            # Create target network
            self.st, self.target_q_values, target_network = self.build_network()
            target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.error, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(q_network_weights)
        if not os.path.exists(self.network_path):
            os.makedirs(self.network_path)


        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_dueling_network(self):
        l_input = Input(shape=(4,84,84))
        conv2d = Conv2D(32,8,strides=(4,4),activation='relu', data_format="channels_first")(l_input)
        conv2d = Conv2D(64,4,strides=(2,2),activation='relu', data_format="channels_first")(conv2d)
        conv2d = Conv2D(64,3,strides=(1,1),activation='relu', data_format="channels_first")(conv2d)
        fltn = Flatten()(conv2d)
        v = Dense(512, activation='relu')(fltn)
        v = Dense(1)(v)
        adv = Dense(512, activation='relu')(fltn)
        adv = Dense(self.num_actions)(adv)
        y = concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(self.num_actions,))(y)
        model = Model(input=l_input,output=l_output)

        s = tf.placeholder(tf.float32, [None, self.state_length, self.frame_width, self.frame_height])
        q_values = model(s)
        return s, q_values, model

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, 8, strides=(4, 4), activation='relu', data_format="channels_first", input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Conv2D(64, 4, strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, 3, strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, self.state_length, self.frame_width, self.frame_height])
        q_values = model(s)
        return s, q_values, model

    def huber_loss(self, x, delta=1.0):
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5,
            delta * (tf.abs(x) - 0.5 * delta)
        )

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        #w = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        # shape = (BATCH_SIZE,)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        td_error = y - q_value
        errors = self.huber_loss(td_error)
        loss = tf.reduce_mean(errors)
        # error_is = (w / tf.reduce_max(w)) * error

        optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=0.95, epsilon=0.01)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, tf.abs(td_error), loss, grad_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (self.frame_width, self.frame_height)) * 255)
        state = [processed_observation for _ in range(self.state_length)]
        return np.stack(state, axis=0)

    def get_action(self, state):
        action = self.repeated_action

        if self.t % self.action_interval == 0:
            if self.epsilon >= random.random() or self.t < self.initial_replay_size:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        # Anneal epsilon linearly over time
        if self.epsilon > self.final_epsilon and self.t >= self.initial_replay_size:
            self.epsilon -= self.anealing_by_step

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(state[1:, :, :], observation, axis=0)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        raw_reward = reward
        reward = np.sign(reward)

        if (not self.prioritized) and len(self.memory) > NUM_REPLAY_MEMORY:
            self.memory.popleft()

        #if self.t < INITIAL_REPLAY_SIZE:
        #self.memory.add(1, (state, action, reward, next_state, terminal))
        self.buffer.append((state, action, reward, next_state, terminal))

        self.R = (self.R + reward * self.gamma_n) / self.gamma

        #print(self.memory.tree.tree[199:])
        #print(self.memory.max_p)
        if self.t < self.initial_replay_size:
            if terminal:      # terminal state
                while len(self.buffer) > 0:
                    n = len(self.buffer)
                    s, a, r, s_, done= self.get_sample(n)
                    if self.prioritized:
                        self.memory.add_p(self.memory.max_p, (s, a, r, s_, done))
                    else:
                        self.memory.append((s, a, r, s_, done))
                    self.R = (self.R - self.buffer[0][2]) / self.gamma
                    self.buffer.pop(0)
                self.R = 0

            if len(self.buffer) >= self.n_step:
                s, a, r, s_, done = self.get_sample(self.n_step)
                if self.prioritized:
                    self.memory.add_p(self.memory.max_p, (s, a, r, s_, done))
                else:
                    self.memory.append((s, a, r, s_, done))
                self.R = self.R - self.buffer[0][2]
                self.buffer.pop(0)



        if self.t >= self.initial_replay_size:

            if terminal:      # terminal state
                while len(self.buffer) > 0:
                    n = len(self.buffer)
                    s, a, r, s_, done= self.get_sample(n)
                    if self.prioritized:
                        self.memory.add_p(self.memory.max_p, (s, a, r, s_, done))
                    else:
                        self.memory.append((s, a, r, s_, done))
                    self.R = (self.R - self.buffer[0][2]) / self.gamma
                    self.buffer.pop(0)
                self.R = 0

            if len(self.buffer) >= self.n_step:
                s, a, r, s_, done = self.get_sample(self.n_step)
                if self.prioritized:
                    self.memory.add_p(self.memory.max_p, (s, a, r, s_, done))
                else:
                    self.memory.append((s, a, r, s_, done))
                self.R = self.R - self.buffer[0][2]
                self.buffer.pop(0)

            # Train network
            if self.t % self.train_interval == 0:
                self.train_network()

            # Update target network
            if self.t % self.target_update_interval == 0:
                print("update",len(self.sess.run(self.update_target_network)))

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.saver.save(self.sess, self.network_path, global_step=(self.t))
                print('Successfully saved: ' + save_path)

        self.total_reward += raw_reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1


        if terminal:
            elapsed = time.time() - self.start
            if self.t < self.initial_replay_size:
                mode = 'random'
            elif self.initial_replay_size <= self.t < self.initial_replay_size + self.anealing_steps:
                mode = 'explore'
            else:
                mode = 'exploit'


            text = 'EPISODE: {0:6d} / TOTAL_STEPS: {1:8d} / STEPS: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / MAX_Q_AVG: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7} / STEPS_PER_SECOND: {8:.1f}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                float(self.total_loss / (self.duration / self.train_interval)), mode, self.duration/elapsed)

            print(text)
            with open(self.env_name+'_output.txt','a') as f:
                f.write(text)

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1
            # Annealing beta
            # self.beta += self.beta_step

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        w_batch = []

        if self.prioritized:
            minibatch, idx_batch = self.memory.sample(self.batch_size)
        else:
            minibatch = random.sample(self.memory, self.batch_size)

        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            #shape = (BATCH_SIZE, 4, 32, 32)
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})

        # DDQN
        if self.double:
            actions = np.argmax(self.q_values.eval(feed_dict={self.s: np.float32(np.array(next_state_batch) / 255.0)}), axis=1)
            target_q_values_batch = np.array([target_q_values_batch[i][action] for i, action in enumerate(actions)])
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma_n * target_q_values_batch
        else:
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=1)

        # IS weight
        #for idx in idx_batch:
        #    wi = (NUM_REPLAY_MEMORY * self.memory.tree.tree[idx])**(-self.beta)
        #    w_batch.append(wi)

        error_batch = self.error.eval(feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        #error_is_batch = self.error_is.eval(feed_dict={
        #   self.a: action_batch,
        #    self.y: y_batch,
        #    self.w: w_batch
        #})

        # Memory update
        if self.prioritized:
            for i in range(self.batch_size):
                self.memory.update(idx_batch[i],error_batch[i])

        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
            #self.w: w_batch
        })


        self.total_loss += loss

    def get_sample(self, n):
        s, a, _, _, _ = self.buffer[0]
        _, _, _, s_, done = self.buffer[n-1]

        return s, a, self.R, s_, done




    def get_action_at_test(self, state):
        action = self.repeated_action

        if self.t % self.action_interval == 0:
            if random.random() <= 0.05:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        self.t += 1

        return action

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.network_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')



    def preprocess(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (self.frame_width, self.frame_height)) * 255)
        return np.reshape(processed_observation, (1, self.frame_width, self.frame_height))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prioritized', type=int, default=1, help='prioritized experience replay')
    parser.add_argument('--double', type=int, default=1, help='Double-DQN')
    parser.add_argument('--dueling', type=int, default=1, help='Dueling Network')
    parser.add_argument('--n_step', type=int, default=3, help='n step bootstrap target')
    parser.add_argument('--env_name', type=str, default='Alien-v0', help='Environment of Atari2600 games')
    parser.add_argument('--train', type=int, default=1, help='train mode or test mode')
    parser.add_argument('--gui', type=int, default=0, help='decide whether you use GUI or not')
    parser.add_argument('--load', type=int, default=0, help='loading saved network')
    parser.add_argument('--network_path', type=str, default=0, help='used in loading and saving (default: \'saved_networks/<env_name>\')')
    parser.add_argument('--replay_memory_size', type=int, default=1000000, help='replay memory size')
    parser.add_argument('--initial_replay_size', type=int, default=20000, help='Learner waits until replay memory stores this number of transition')
    parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes each agent plays')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')

    args = parser.parse_args()

    if args.network_path == 0:
        args.network_path = 'saved_networks/' + args.env_name
    if not os.path.exists(args.network_path):
        os.makedirs(args.network_path)



    env = gym.make(args.env_name)
    #env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1',discrete_actions=True)
    agent = Agent(args, num_actions=env.action_space.n)

    if args.train:  # Train mode
        for _ in range(args.num_episodes):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, agent.no_op_steps)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            agent.start = time.time()
            while not terminal:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                if args.gui:
                    env.render()
                processed_observation = agent.preprocess(observation, last_observation)
                state = agent.run(state, action, reward, terminal, processed_observation)
                # Test mode
                # env.monitor.start(ENV_NAME + '-test')

    env = wrappers.Monitor(env, args.network_path, force=True)
    for _ in range(10):
        terminal = False
        observation = env.reset()
        for _ in range(random.randint(1, agent.no_op_steps)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing
        state = agent.get_initial_state(observation, last_observation)
        while not terminal:
            last_observation = observation
            action = agent.get_action_at_test(state)
            observation, _, terminal, _ = env.step(action)
            env.render()
            processed_observation = preprocess(observation, last_observation)
            state =np.append(state[1:, :, :], processed_observation, axis=0)
            # env.monitor.close()


if __name__ == '__main__':
    main()

