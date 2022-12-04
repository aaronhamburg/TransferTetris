from matris import Game, GameOver, WIDTH, HEIGHT, MATRIX_WIDTH, MATRIX_HEIGHT
import time
import random

import numpy as np
# import gym
# from gym import wrappers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pygame

DRAW_WAIT_TIME = 0.1

class Agent(object):
    
    def __init__(self):
        self.n_actions = 44
        # we define some parameters and hyperparameters:
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005
        self.batch_size = 32
        
        # We define our memory buffer where we will store our experiences
        # We stores only the 2000 last time steps
        self.memory_buffer= list()
        self.max_memory_buffer = 2000
        
        # We create our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space

        self.model = Sequential([
            Dense(units=24,input_dim=MATRIX_WIDTH * (MATRIX_HEIGHT + 4), activation = 'relu'),
            Dense(units=24,activation = 'relu'),
            Dense(units=self.n_actions, activation = 'linear')
        ])
        self.model.compile(loss="mse",
                      optimizer = Adam(learning_rate=self.lr))

    def run_episode(self, draw_screen=False):
        game = Game()
        game.main(draw_screen=draw_screen, comp_control=True)
        next_state = None
        current_state = None
        action = None
        reward = None
        steps = 0
        while True:
            try:
                if next_state is not None:
                    current_state = next_state
                else:
                    current_state = game.matris.current_state()

                steps += 1
                action = self.compute_action(current_state)
                rotation, position = self.action_to_tuple(action)
                if draw_screen:
                    time.sleep(DRAW_WAIT_TIME)
                reward = game.matris.computer_update(rotation, position)
                next_state = game.matris.current_state()
                # return whether done and next state, maybe can set next_state = current state at the start of the loop if it's set

                # We store each experience in the memory buffer
                self.store_episode(current_state.reshape((1,260)), action, reward, next_state.reshape((1,260)), False)

                if draw_screen: 
                    time.sleep(DRAW_WAIT_TIME)
                    game.redraw()
            except GameOver:
                next_state = game.matris.current_state()
                self.store_episode(current_state.reshape((1,260)), action, 0, next_state.reshape((1,260)), True)
                self.update_exploration_probability()
                if draw_screen:
                    time.sleep(DRAW_WAIT_TIME)
                    pygame.display.quit()
                return steps


    # The agent computes the action to perform given a state 
    def compute_action(self, current_state):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action 
        #     with the highest Q-value.
        if np.random.uniform(0,1) < self.exploration_proba:
            return np.random.choice(range(self.n_actions))
        q_values = self.model.predict(current_state.reshape((1,260)))[0]
        return np.argmax(q_values)


    def action_to_tuple(self, action):
        rotation = action % 4
        position = action % 11 - 2
        return (rotation, position)
        


    # when an episode is finished, we update the exploration probability using 
    # espilon greedy algorithm
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        print(self.exploration_proba)
    
    # At each time step, we store the corresponding experience
    def store_episode(self,current_state, action, reward, next_state, done):
        #We use a dictionnary to store them
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    

    # At the end of each episode, we train our model
    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]
        
        # We iterate over the selected experiences
        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.model.predict(experience["current_state"])
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma*np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            # train the model
            self.model.fit(experience["current_state"], q_current_state, verbose=0)

if __name__ == '__main__':

    # We create our gym environment 
    # env = gym.make("CartPole-v1")
    
    # Number of episodes to run
    n_episodes = 400
    draw_frequency = 50
    # We define our agent
    agent = Agent()

    for ep in range(n_episodes):
        
        total_steps = 0

        total_steps += agent.run_episode(draw_screen=ep % draw_frequency == draw_frequency - 1)

        if total_steps >= agent.batch_size:
            print("reached agent.train")
            agent.train()


    