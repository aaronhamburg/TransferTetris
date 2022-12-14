from matris import Game, GameOver, WIDTH, HEIGHT, MATRIX_WIDTH, MATRIX_HEIGHT
import time
# import random

import numpy as np
# import gym
# from gym import wrappers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import pygame
import gc

DRAW_WAIT_TIME = .025

class Agent(object):
    
    def __init__(self):
        self.n_actions = 5
        # we define some parameters and hyperparameters:
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN
        self.lr = 0.01
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005
        self.batch_size = 64
        
        # We define our memory buffer where we will store our experiences
        # We stores only the 2000 last time steps
        self.memory_buffer= list()
        self.max_memory_buffer = 2000
        
        # We create our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space

        # self.model = Sequential([
        #     Dense(units=24,input_dim=MATRIX_WIDTH * (MATRIX_HEIGHT + 4), activation = 'relu'),
        #     Dense(units=24,activation = 'relu'),
        #     Dense(units=self.n_actions, activation = 'linear')
        # ])
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(MATRIX_HEIGHT, MATRIX_WIDTH, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.n_actions, activation='linear'))
        self.model = model
        self.model.compile(loss="mse",
                      optimizer = Adam(learning_rate=self.lr))

    def run_episode(self, draw_screen=False, always_explore=False, max_steps=-1):
        game = Game()
        game.main(draw_screen=draw_screen, comp_control=True)
        next_state = None
        current_state = None
        action = None
        reward = None
        steps = 0
        total_score = 0
        total_reward = 0
        condition = True
        if max_steps >= 0:
            condition = steps < max_steps
        while condition:
            try:
                if next_state is not None:
                    current_state = next_state
                else:
                    current_state = game.matris.current_state()

                steps += 1
                action = self.compute_action(current_state, always_explore=always_explore)
                action_string = self.action_to_move(action)
                if draw_screen:
                    time.sleep(DRAW_WAIT_TIME)
                score, reward = game.matris.computer_update(action_string)
                total_score += score
                total_reward += reward
                next_state = game.matris.current_state()
                # return whether done and next state, maybe can set next_state = current state at the start of the loop if it's set

                # We store each experience in the memory buffer
                self.store_episode(current_state, action, reward, next_state, False)

                if draw_screen: 
                    time.sleep(DRAW_WAIT_TIME)
                    game.redraw()
            except GameOver:
                self.store_episode(current_state, action, -1000, None, True)
                if not always_explore:
                    self.update_exploration_probability()
                if draw_screen:
                    time.sleep(DRAW_WAIT_TIME)
                    pygame.display.quit()
                return (steps, total_score, total_reward, game.matris.lines)


    # The agent computes the action to perform given a state 
    def compute_action(self, current_state, always_explore=False):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action 
        #     with the highest Q-value.
        if always_explore or np.random.uniform(0,1) < self.exploration_proba:
            return np.random.choice(range(self.n_actions))

        tensor = tf.convert_to_tensor(np.array([current_state]), dtype=tf.float32)
        q_values = self.model.predict(tensor, verbose=0)[0]
        return np.argmax(q_values)


    def action_to_move(self, action):
        if action == 0: 
            return 'left'
        if action == 1:
            return 'right'
        if action == 2:
            return 'rotate'
        if action == 3:
            return 'drop'
        return 'do nothing'
        
        


    # when an episode is finished, we update the exploration probability using 
    # espilon greedy algorithm
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        # print(self.exploration_proba)

    # At each time step, we store the corresponding experience
    def store_episode(self,current_state, action, reward, next_state, done):
        #We use an Experience object to store them so they can be sorted into a max heap
        # self.memory_buffer.append(self.Experience(np.array([current_state]), action, reward, np.array([next_state]), done))
        self.memory_buffer.append({
            "current_state":np.array([current_state]),
            "action":action,
            "reward":reward,
            "next_state":np.array([next_state]),
            "done":done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    

    # At the end of each episode, we train our model
    def train(self):
        K.clear_session()
        gc.collect()
        # # We shuffle the memory buffer and select a batch size of experiences
        batch = np.random.permutation(self.memory_buffer)

        # We choose the elements with the highest absolute value reward to encourage it to learn from important events
        batch_sample = batch[0:self.batch_size]
        q_targets = np.zeros((self.batch_size, self.n_actions))
        states = np.zeros((self.batch_size,) + (MATRIX_HEIGHT, MATRIX_WIDTH, 1))

        # We iterate over the selected experiences
        for index, experience in enumerate(batch_sample):
            # We compute the Q-values of S_t
            tensor = tf.convert_to_tensor(experience['current_state'], dtype=tf.float32)
            q_current_state = self.model.predict(tensor, verbose=0)
            # We compute the Q-target using Bellman optimality equation
            q_target = experience['reward']
            if not experience['done']:
                tensor = tf.convert_to_tensor(experience['next_state'], dtype=tf.float32)
                q_target = q_target + self.gamma*np.max(self.model.predict(tensor, verbose=0)[0])
            q_current_state[0][experience['action']] = q_target
            q_targets[index] = q_current_state[0]
            states[index] = experience['current_state']
            # train the model
            # self.model.fit(experience['current_state'], q_current_state, verbose=0)

        self.model.fit(states, q_targets, verbose=0)

if __name__ == '__main__':

    # We create our gym environment 
    # env = gym.make("CartPole-v1")
    
    # Number of episodes to run
    n_explore_episodes = 0
    n_episodes = 500
    
    # After how many episodes it should be drawn (if 0 it will never draw)
    draw_frequency = 0

    total_steps = 0
    scores = np.zeros(n_episodes)
    rewards = np.zeros(n_episodes)
    # We define our agent
    agent = Agent()

    for ep in range(n_episodes):
        
        if ep % 10 == 0:
            print("Episode " + str(ep + 1))
        

        explore = ep < n_explore_episodes

        draw = False
        if not draw_frequency == 0:
            draw = ep % draw_frequency == draw_frequency - 1
        steps, score, reward = agent.run_episode(draw_screen=draw and not explore, always_explore=explore)
        total_steps += steps
        scores[ep] = score
        rewards[ep] = reward

        if total_steps >= agent.batch_size:
            agent.train()
    

    plt.plot(scores)
    plt.xlabel("Episode #")
    plt.ylabel("Total score")
    plt.savefig('heuristic_square_scores.png')
    plt.plot(rewards)
    plt.xlabel("Episode #")
    plt.ylabel("Total reward (heuristic)")
    plt.savefig('heuristic_square_rewards.png')


    