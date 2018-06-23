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
import threading

ENV_NAME = 'Breakout-v0'  # Environment name
TRAIN = True
LOAD_NETWORK = False
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME

NUM_ACTORS = 1
NUM_EPISODES = 12000  # Number of episodes the agent plays
INITIAL_REPLAY_SIZE = 50000  # The Learner awaits for this size of transitions to be accumulated.
NUM_REPLAY_MEMORY = 200000  # Remote memory size
MEMORY_REMOVE_INTERVAL = 100
PARAMETER_COPY_INTERVAL = 400
EPSILON_EXPOENT_ALPHA = 7
EPSILON = 0.4
SEND_BATCH_SIZE = 50
PRINT_INTERVAL = 300
N_STEP_RETURN = 3
GAMMA = 0.99  # Discount factor
GAMMA_N = GAMMA ** N_STEP_RETURN
PRIORITY_ALPHA = 0.6

# About epsilon-greedy
ANEALING_EPSILON = True
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy


FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
BATCH_SIZE = 32  # Mini batch size, 512 is the best.
TARGET_UPDATE_INTERVAL = 2500 # The frequency with which the target network is updated
ACTION_INTERVAL = 4  # The agent sees only every () input
LEARNING_RATE = 0.00025 / 4  # Learning rate used by RMSProp
SAVE_INTERVAL = 50000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
NUM_EPISODES_AT_TEST = 10  # Number of episodes the agent plays at test time

class Memory:
    def __init__(self):
        self.transition = deque()
        self.priorities = deque()
        self.total_p = 0

    def _error_to_priority(self, error_batch):
        priority_batch = []
        for error in error_batch:
            priority_batch.append(error**PRIORITY_ALPHA)
        return priority_batch

    def length(self):
        return len(self.transition)

    def add(self, transiton_batch, error_batch):
        priority_batch = self._error_to_priority(error_batch)
        self.total_p += sum(priority_batch)
        self.transition.extend(transiton_batch)
        self.priorities.extend(priority_batch)

    def sample(self, n):
        batch = []
        idx_batch = []
        segment = self.total_p / n

        idx = -1
        sum_p = 0
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            while sum_p < s:
                sum_p += self.priorities[idx]
                idx += 1
            idx_batch.append(idx)
            batch.append(self.transition[idx])
        return batch, idx_batch


    def update(self, idx_batch, error_batch):
        priority_batch = self._error_to_priority(error_batch)
        for i in range(len(idx_batch)):
            change = priority_batch[i] - self.priorities[idx_batch[i]]
            self.total_p += change
            self.priorities[idx_batch[i]] = priority_batch[i]


    def remove(self):
        print("Excess Memory: ", (len(self.priorities) - NUM_REPLAY_MEMORY))
        for _ in range(len(self.priorities) - NUM_REPLAY_MEMORY):
            self.transition.popleft()
            p = self.priorities.popleft()
            self.total_p -= p



class Learner:
    def __init__(self, sess):
        self.sess = sess
        self.f_end = False
        self.env = gym.make(ENV_NAME)

        self.num_actions = self.env.action_space.n

        self.t = 0
        self.total_time = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        self.start = 0

        with tf.variable_scope("learner_parameters", reuse=True):
            self.s, self.q_values, q_network = self.build_network()
        q_network_weights = self.bubble_sort_parameters(q_network.trainable_weights)

        # Create target network
        with tf.variable_scope("learner_target_parameters", reuse=True):
            self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = self.bubble_sort_parameters(target_network.trainable_weights)

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]


        # Define loss and gradient update operation
        self.a, self.y, self.error, self.loss, self.grad_update, self.gv, self.cl = self.build_training_op(q_network_weights)




        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        with tf.device("/cpu:0"):
            self.saver = tf.train.Saver(q_network_weights)

        self.sess.run(tf.global_variables_initializer())

        # Initialize target network
        self.sess.run(self.update_target_network)


    def bubble_sort_parameters(self, arr):
        change = True
        while change:
            change = False
            for i in range(len(arr) - 1):
                if arr[i].name > arr[i + 1].name:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    change = True
        return arr


    def build_network(self):
        l_input = Input(shape=(4,84,84))
        conv2d = Conv2D(32,8,strides=(4,4),activation='relu', data_format="channels_first")(l_input)
        conv2d = Conv2D(64,4,strides=(2,2),activation='relu', data_format="channels_first")(conv2d)
        conv2d = Conv2D(64,3,strides=(1,1),activation='relu', data_format="channels_first")(conv2d)
        fltn = Flatten()(conv2d)
        v = Dense(512, activation='relu', name="dense_v1")(fltn)
        v = Dense(1, name="dense_v2")(v)
        adv = Dense(512, activation='relu', name="dense_adv1")(fltn)
        adv = Dense(self.num_actions, name="dense_adv2")(adv)
        y = concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(self.num_actions,))(y)
        model = Model(input=l_input,output=l_output)

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        # shape = (BATCH_SIZE,)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        # error_is = (w / tf.reduce_max(w)) * error
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.95, epsilon=1.5e-7, centered=True)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=q_network_weights)
        capped_gvs = [(grad if grad is None else tf.clip_by_norm(grad, clip_norm=40), var) for grad, var in grads_and_vars]
        grad_update = optimizer.apply_gradients(capped_gvs)

        return a, y, error, loss, grad_update ,grads_and_vars, capped_gvs

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')



    def run(self):
        global total_episode

        # This should be done after Actors were generated.
        if LOAD_NETWORK:
            self.load_network()

        if remote_memory.length() < INITIAL_REPLAY_SIZE:
            print("Learner Waiting...")
            time.sleep(10)
            self.run()

        if not self.f_end:
            print("Learner Starts!")


        while not self.f_end:
            start = time.time()

            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            terminal_batch = []

            minibatch, idx_batch = remote_memory.sample(BATCH_SIZE)

            for data in minibatch:
                state_batch.append(data[0])
                action_batch.append(data[1])
                reward_batch.append(data[2])
                #shape = (BATCH_SIZE, 4, 32, 32)
                next_state_batch.append(data[3])
                terminal_batch.append(data[4])

                self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(data[0] / 255.0)]},session=self.sess))

            # Convert True to 1, False to 0
            terminal_batch = np.array(terminal_batch) + 0
            # shape = (BATCH_SIZE, num_actions)
            target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)}, session=self.sess)
            # DDQN
            actions = np.argmax(self.q_values.eval(feed_dict={self.s: np.float32(np.array(next_state_batch) / 255.0)}, session=self.sess), axis=1)
            target_q_values_batch = np.array([target_q_values_batch[i][action] for i, action in enumerate(actions)])
            # shape = (BATCH_SIZE,)
            y_batch = reward_batch + (1 - terminal_batch) * GAMMA_N * target_q_values_batch


            error_batch = self.error.eval(feed_dict={
                self.s: np.float32(np.array(state_batch) / 255.0),
                self.a: action_batch,
                self.y: y_batch
            }, session=self.sess)

            loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
                self.s: np.float32(np.array(state_batch) / 255.0),
                self.a: action_batch,
                self.y: y_batch
            })


            self.total_loss += loss
            self.total_time += time.time() - start

            # Memory update
            remote_memory.update(idx_batch, error_batch)

            self.t += 1

            if self.t % PRINT_INTERVAL == 0:
                text_l = 'AVERAGE LOSS: {0:.5F} / AVG_MAX_Q: {1:2.4F} / LEARN PER SECOND: {2:.1F} / NUM LEARN: {3:5d}'.format(
                    self.total_loss/PRINT_INTERVAL, self.total_q_max/(PRINT_INTERVAL*BATCH_SIZE), PRINT_INTERVAL/self.total_time, self.t)
                print(text_l)
                with open(ENV_NAME+'_output.txt','a') as f:
                    f.write(text_l+"\n")
                self.total_loss = 0
                self.total_time = 0
                self.total_q_max = 0

            # Remove excess memory
            if self.t % MEMORY_REMOVE_INTERVAL == 0 and remote_memory.length() > NUM_REPLAY_MEMORY:
                remote_memory.remove()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=(self.t))
                print('Successfully saved: ' + save_path)

            if total_episode >= NUM_EPISODES:
                self.f_end = True

        print("The Learning is Over.")
        time.sleep(0.5)


class Actor:
    def __init__(self, number, sess):
        self.sess = sess
        self.f_end = False

        self.env = gym.make(ENV_NAME)

        self.num = number
        self.num_actions = self.env.action_space.n
        self.t = 0
        self.repeated_action = 0

        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        if NUM_ACTORS != 1:
            self.epsilon = EPSILON **(1+(self.num/(NUM_ACTORS-1))*EPSILON_EXPOENT_ALPHA)
        else:
            self.epsilon = EPSILON


        if ANEALING_EPSILON:
            self.epsilon = INITIAL_EPSILON
            self.epsilon_step = (INITIAL_EPSILON -FINAL_EPSILON)/ EXPLORATION_STEPS


        self.local_memory = deque(maxlen=100)
        self.buffer = []
        self.R = 0

        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = self.bubble_sort_parameters(q_network.trainable_weights)

        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = self.bubble_sort_parameters(target_network.trainable_weights)

        q_parameters = self.bubble_sort_parameters(tf.trainable_variables(scope="learner_parameters"))
        target_parameters = self.bubble_sort_parameters(tf.trainable_variables(scope="learner_target_parameters"))

        self.obtain_q_parameters = [q_network_weights[i].assign(q_parameters[i]) for i in range(len(q_parameters))]
        self.obtain_target_parameters = [target_network_weights[i].assign(target_parameters[i]) for i in range(len(target_parameters))]

        self.a, self.y, self.q, self.error = self.td_error_op()

        self.sess.run(tf.global_variables_initializer())


    def bubble_sort_parameters(self, arr):
        change = True
        while change:
            change = False
            for i in range(len(arr) - 1):
                if arr[i].name > arr[i + 1].name:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    change = True
        return arr


    def td_error_op(self):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        q = tf.placeholder(tf.float32, [None,None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        # shape = (BATCH_SIZE,)
        q_value = tf.reduce_sum(tf.multiply(q, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)

        return a, y, q, error


    def build_network(self):
        l_input = Input(shape=(4,84,84))
        conv2d = Conv2D(32,8,strides=(4,4),activation='relu', data_format="channels_first")(l_input)
        conv2d = Conv2D(64,4,strides=(2,2),activation='relu', data_format="channels_first")(conv2d)
        conv2d = Conv2D(64,3,strides=(1,1),activation='relu', data_format="channels_first")(conv2d)
        fltn = Flatten()(conv2d)
        v = Dense(512, activation='relu', name="dense_v1_"+str(self.num))(fltn)
        v = Dense(1, name="dense_v2_"+str(self.num))(v)
        adv = Dense(512, activation='relu', name="dense_adv1_"+str(self.num))(fltn)
        adv = Dense(self.num_actions, name="dense_adv2_"+str(self.num))(adv)
        y = concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(self.num_actions,))(y)
        model = Model(input=l_input,output=l_output)

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)


    def preprocess(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))



    def get_action_and_q(self, state):
        action = self.repeated_action
        q = self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}, session=self.sess)
        if self.t % ACTION_INTERVAL == 0:
            if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(q[0])
            self.repeated_action = action
        return action, q[0]

    def get_action_at_test(self, state):
        action = self.repeated_action

        if self.t % ACTION_INTERVAL == 0:
            if random.random() <= 0.05:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        self.t += 1

        return action

    def get_sample(self, n):
        s, a, _, _, _, q = self.buffer[0]
        _, _, _, s_, done, q_ = self.buffer[n-1]

        return s, a, self.R, s_, done, q, q_


    def run(self):
        global total_episode

        if TRAIN:  # Train mode
            while not self.f_end:
                terminal = False
                observation = self.env.reset()
                for _ in range(random.randint(1, NO_OP_STEPS)):
                    last_observation = observation
                    observation, _, _, _ = self.env.step(0)  # Do nothing
                state = self.get_initial_state(observation, last_observation)
                start = time.time()
                while not terminal:
                    last_observation = observation
                    action, q = self.get_action_and_q(state)
                    observation, reward, terminal, _ = self.env.step(action)
                    reward = np.sign(reward)
                    #env.render()
                    processed_observation = self.preprocess(observation, last_observation)
                    next_state = np.append(state[1:, :, :], processed_observation, axis=0)


                    self.buffer.append((state, action, reward, next_state, terminal, q))
                    self.R = (self.R + reward * GAMMA_N) / GAMMA

                    # n-step transition
                    if terminal:      # terminal state
                        while len(self.buffer) > 0:
                            n = len(self.buffer)
                            s, a, r, s_, done, q, q_ =  self.get_sample(n)
                            self.local_memory.append((s, a, r, s_, done, q, q_))
                            self.R = (self.R - self.buffer[0][2]) / GAMMA
                            self.buffer.pop(0)
                        self.R = 0

                    if len(self.buffer) >= N_STEP_RETURN:
                        s, a, r, s_, done, q, q_ = self.get_sample(N_STEP_RETURN)
                        self.local_memory.append((s, a, r, s_, done, q, q_))
                        self.R = self.R - self.buffer[0][2]
                        self.buffer.pop(0)

                    # Add experience and priority to remote memory
                    if len(self.local_memory) > 50:
                        state_batch = []
                        action_batch = []
                        reward_batch = []
                        next_state_batch = []
                        terminal_batch = []
                        q_batch = []
                        qn_batch = []

                        for _ in range(SEND_BATCH_SIZE):
                            data = self.local_memory.popleft()
                            state_batch.append(data[0])
                            action_batch.append(data[1])
                            reward_batch.append(data[2])
                            #shape = (BATCH_SIZE, 4, 32, 32)
                            next_state_batch.append(data[3])
                            terminal_batch.append(data[4])
                            q_batch.append(data[5])
                            qn_batch.append(data[6])

                        terminal_batch = np.array(terminal_batch) + 0
                        # shape = (BATCH_SIZE, num_actions)
                        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)}, session=self.sess)
                        # DDQN
                        actions = np.argmax(qn_batch, axis=1)
                        target_q_values_batch = np.array([target_q_values_batch[i][action] for i, action in enumerate(actions)])
                        # shape = (BATCH_SIZE,)
                        y_batch = reward_batch + (1 - terminal_batch) * GAMMA_N * target_q_values_batch

                        error_batch = self.error.eval(feed_dict={
                            self.s: np.float32(np.array(state_batch) / 255.0),
                            self.a: action_batch,
                            self.q: q_batch,
                            self.y: y_batch
                        }, session=self.sess)

                        send = [(state_batch[i],action_batch[i],reward_batch[i],next_state_batch[i],terminal_batch[i]) for i in range(SEND_BATCH_SIZE)]

                        remote_memory.add(send, error_batch)

                    state = next_state

                    self.t += 1

                    if self.t % PARAMETER_COPY_INTERVAL == 0:
                        self.sess.run(self.obtain_q_parameters)
                        self.sess.run(self.obtain_target_parameters)

                    if ANEALING_EPSILON and EXPLORATION_STEPS + INITIAL_REPLAY_SIZE > self.t >= INITIAL_REPLAY_SIZE:
                        self.epsilon -= self.epsilon_step

                    self.total_reward += reward
                    self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}, session=self.sess))
                    self.duration += 1

                elapsed = time.time() - start

                text = 'EPISODE: {0:6d} / ACTOR: {1:3d} / TIMESTEP: {2:8d} / DURATION: {3:5d} / EPSILON: {4:.5f} / TOTAL_REWARD: {5:3.0f} / AVG_MAX_Q: {6:2.4f} / STEP_PER_SECOND: {7:.1f}'.format(
                    self.episode + 1, self.num, self.t, self.duration, self.epsilon,
                    self.total_reward, self.total_q_max / float(self.duration),
                    self.duration/elapsed)

                print(text)


                with open(ENV_NAME+'_output.txt','a') as f:
                    f.write(text+"\n")

                self.total_reward = 0
                self.total_q_max = 0
                self.total_loss = 0
                self.duration = 0
                self.episode += 1

                total_episode += 1
                if total_episode >= NUM_EPISODES:
                    self.f_end = True

            print("Actor",self.num,"is Over.")
            time.sleep(0.5)


total_episode = 0
remote_memory = Memory()

# Train Mode
if TRAIN:
    sess = tf.InteractiveSession()
    #with tf.device("/gpu:0"):
    threads = [Learner(sess)]
    #with tf.device("/cpu:0"):
    for i in range(NUM_ACTORS):
        threads.append(Actor(number=i, sess=sess))

    jobs = []
    for worker in threads:
        job = lambda: worker.run()
        t = threading.Thread(target=job)
        jobs.append(t)
        t.start()


    for t in jobs:
        t.join()

# Test Mode
else:
    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, SAVE_NETWORK_PATH, force=True)
    sess = tf.InteractiveSession()
    leaner = Learner(sess)
    agent = Actor(number=0,sess=sess)
    leaner.load_network()
    agent.sess.run(agent.obtain_q_parameters)
    for _ in range(NUM_EPISODES_AT_TEST):
        terminal = False
        observation = env.reset()
        for _ in range(random.randint(1, NO_OP_STEPS)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing
        state = agent.get_initial_state(observation, last_observation)
        while not terminal:
            last_observation = observation
            action = agent.get_action_at_test(state)
            observation, _, terminal, _ = env.step(action)
            env.render()
            processed_observation = agent.preprocess(observation, last_observation)
            state =np.append(state[1:, :, :], processed_observation, axis=0)



