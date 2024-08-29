from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
import os

from memory import ReplayMemory
from models import NoisyDQN,DQN  # Adjust based on your actual model definitions
from game_small_maze import Maze

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import data
# import levenshtein_distance
# import optics


Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

# Constants
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 370 
CELL_SIZE = 20
MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = (SCREEN_HEIGHT - 50) // CELL_SIZE  # Adjusted height for timer display

reward_list = []
def get_state(obs, episode, save_image=False):
    map_info = obs['maze'] 
    #-->地圖資訊
    map_array = np.array(map_info).reshape((MAZE_HEIGHT, MAZE_WIDTH))
    map_array = np.resize(map_array, (32, 32))

    normalized_map = map_array / np.max(map_array)
    
    state = torch.tensor(normalized_map, dtype=torch.float).unsqueeze(0) 
    state = state.unsqueeze(0) 

    if save_image:
        plt.imshow(normalized_map, cmap='gray')
        filename = f'./temp/{episode}/{steps_done}.png'
        dir = os.path.dirname(filename)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(filename)
        plt.close()
    
    return state

def test(env, n_episodes, policy, render=True, device='cpu'):
    path_array = []
    global steps_done
    policy.to(device)  # Move policy network to the specified device

    for episode in range(n_episodes):
        path_array.append(data.dat())
        obs = env.reset()
        state = get_state(obs, episode, False).to(device)  # Ensure state is on the same device as the policy
        path_array[episode].AddState(state.float())
        total_reward = 0.0
        #obs['maze']可以拿到地圖資訊
        for t in count():
            action = policy(state).max(1)[1].view(1, 1)
            path_array[episode].AddAction(action)
            steps_done += 1
            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done = env.step(action.item())

            total_reward += reward

            if not done:
                next_state = get_state(obs, episode, False).to(device)  # Ensure next_state is on the same device as the policy
                path_array[episode].AddState(next_state.float())
            else:
                next_state = None

            state = next_state
            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    print("-------------------------------\n L =", len(path_array[0].get_state()))  # Print state length
    return path_array

if __name__ == '__main__':
    tt = False
    # if tt:
    #     optics.optic().clustering_from_mem()
    #     exit()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Maze(countdown_time=30,rendering_enabled=True)

    n_actions = 4
    policy_net = DQN(n_actions=n_actions).to(device)
    steps_done = 0

    # Load model
    policy_net.load_state_dict(torch.load("DQN_Smaze_20001", map_location=device))
    path_array = test(env, 40, policy_net, render=False)
    
    # dis_graph = levenshtein_distance.PathDistanceCalculator().calculate_distances(path_array)
    # print(dis_graph)
    # print("----------------------------------------------")
    
    # # Save distance graph
    # with open('test.mem', 'w') as f:
    #     for i in dis_graph:
    #         for j in i:
    #             f.write(str(int(j)) + " ")
    #         f.write("\n")
    
    # optics.optic().clustering(dis_graph)
