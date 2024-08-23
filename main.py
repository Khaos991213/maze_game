import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time
from PIL import Image

import matplotlib.pyplot as plt
import os
from memory import ReplayMemory
from models import DQN, DQNbn  # Adjust based on your actual model definitions

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import data
# import levenshtein_distance
# import optics
from game import Maze, Player, Timer

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 650  # Increased height for timer display
CELL_SIZE = 20
MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = (SCREEN_HEIGHT - 50) // CELL_SIZE  # Adjusted height for timer display
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (192, 192, 192)
BLUE = (0, 0, 255)

reward_list = []
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device).view(-1, 1)  # Ensure shape [BATCH_SIZE, 1]
    reward_batch = torch.cat(batch.reward).to(device)

    # print("state_batch shape:", state_batch.shape)
    # print("action_batch shape:", action_batch.shape)

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_action_values = policy_net(state_batch)
    # print("state_action_values shape:", state_action_values.shape)
    
    # Ensure action_batch is correctly shaped
    state_action_values = state_action_values.gather(1, action_batch)  # Gather the action values

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# def get_state(obs, ep, p):
#     state = np.array(obs)
#     print("Shape of obs:", np.array(obs).shape)

#     if p:
#         plt.imshow(state)
#         filename = f'./temp/{ep}/{steps_done}.png'
#         dir = os.path.dirname(filename)
#         if dir and not os.path.exists(dir):
#             os.makedirs(dir)
#         plt.savefig(filename)
#         plt.close()
#     state = state.transpose((2, 0, 1))
#     state = torch.from_numpy(state)
#     return state.unsqueeze(0)

def get_screen_rgb(env):
    screen = env.get_screen_rgb( )
    # Convert numpy array to PIL Image
    screen = Image.fromarray(screen)

    transform = T.Compose([
        T.Resize((84, 84)),  
        T.ToTensor(),  
    ])
    
    screen = transform(screen)

    return screen.unsqueeze(0).to(device)

def get_state(obs, episode, save_image=False):
    # Extract the map information from the dictionary
    map_info = obs['maze']  # Assuming 'maze' key contains the map data
    # Convert the 1D list to a 2D array (assuming you have MAZE_WIDTH and MAZE_HEIGHT defined)
    map_array = np.array(map_info).reshape((MAZE_HEIGHT, MAZE_WIDTH))
    
    map_array = np.resize(map_array, (32, 32))  # Adjust this if necessary

    # Normalize the array to range [0, 1] if it's not already normalized
    normalized_map = map_array / np.max(map_array)
    
    # Convert to a tensor
    state = torch.tensor(normalized_map, dtype=torch.float).unsqueeze(0)  # Add batch dimension
    state = state.unsqueeze(0)  # Add channel dimension

    # Optional: Save the state as an image for visualization/debugging
    if save_image:
        plt.imshow(normalized_map, cmap='gray')  # Use 'gray' colormap for simplicity
        filename = f'./temp/{episode}/{steps_done}.png'
        dir = os.path.dirname(filename)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(filename)
        plt.close()
    
    return state


def train(env, n_episodes, render=False):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs, episode, False)
        total_reward = 0.0
        for t in count():
            
            action = select_action(state)

            if render:
                env.render()

            obs, reward, done= env.step(action)  # Adjusted for 4 values return
            
            total_reward += reward

            if not done:
                # next_state = get_screen_rgb(env) if not done else None
                next_state = get_state(obs, episode, False)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cuda'), next_state, reward.to('cuda'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
    env.close()

def test(env, n_episodes, policy, render=True):
    path_array = []
    global steps_done
    for episode in range(n_episodes):
        path_array.append(data.dat())
        obs = env.reset()
        state = get_state(obs[0], episode, False)
        path_array[episode].AddState(state.float())
        total_reward = 0.0
        for t in count():
            action = policy(state).max(1)[1].view(1, 1)
            path_array[episode].AddAction(action)
            steps_done += 1
            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, _ = env.step(action.item())  # Ensure action is a single item integer

            total_reward += reward

            if not done:
                next_state = get_state(obs, episode, False)
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

    # Hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    
    # Create environment
    env = Maze(countdown_time=60)
    
    # Define the number of actions in your environment
    n_actions = 4  # Adjust this based on your environment's action space

    # Create networks
    policy_net = DQN(n_actions=n_actions).to(device)
    target_net = DQN(n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    steps_done = 0

    # Initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Train model
    train(env, 1001)
    
    torch.save(policy_net.state_dict(), "dqn_alien_model_30001")
    
    # Load model
    # policy_net.load_state_dict(torch.load("dqn_alien_model_30001", map_location=device))
    # path_array = test(env, 40, policy_net, render=False)
    
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
