import torch
import torch.nn as nn
import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from tqdm import tqdm
import pickle
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym
import numpy as np
import collections
import cv2
import matplotlib.pyplot as plt


########################################################
# environment classes and setup

# functions for creating the function
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        '''return only every "skip"-th frame'''
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._observation_buffer = collections.deque(maxlen=2)
        self._skip = skip

        def step(self, action):
            pass

        def reset(self):
            '''clear buffer and init to first obs'''
            self._observation_buffer.clear()  # empties deque
            obs = self.env.reset()
            self._observation_buffer.append(obs)
            return obs


class ProcessFrame84(gym.ObservationWrapper):
    '''Dornsamples image to 84x84, greyscales image, returns numpy array'''

    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."

        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        resized_screen = cvs.resze(
            img, (84, 110)
            ,interpolation=cv2.INTER_AREA
        )
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
    '''makes the image accessible by pytorch'''
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low = 0.0
            ,high = 1.0
            ,shape = (old_shape[-1], old_shape[0], old_shape[1])
            ,dtype = np.float32
        )

        def observation(self, observation):
            return np.moveaxis(observation_space, 2, 0)

class BufferWrapper(gym.ObservationWrapper):
    '''wraps environment space into buffer steps'''
    def __init__(self, env, n_steps, dtype = np.float32):
        super().__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0)
            ,old_space.high.repeat(n_steps, axis=0)
            ,dtype = dtype
        )

        def reset(self):
            self.buffer = np.zeros_like(self.observation_space.low, dtype = self.dtype)
            return self.observation(self.env.reset())

        def observation(self, observation):
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = observation
            return self.buffer

class ScaledFloatFrame(gym.ObservationWrapper):
    '''normalize pixel values in frame -> 0 to 1'''
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0



def make_env(game_name):
    '''takes game version and creates environment from gym_super_mario_bros library'''
    env = gym_super_mario_bros.make(game_name)
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)

    return JoypadSpace(env, RIGHT_ONLY)


########################################################
# Double Deep Q Network code

class DQNSolver(nn.Module):

    def __init__(self, input_shape, n_actions):
        super().__init__()

        # setting up convultion network
        # applying ReLU to nodes and changing node sizes
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernal_size = 8, stride = 4)
            ,nn.ReLU()
            ,nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
            ,nn.ReLU()
            ,nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
            ,nn.ReLU()
        )

        # getting final convolution network size
        conv_out_size = self._get_conv_out(input_shape)
        # applying sequential linear to 512 (nodes?), ReLU
        # ,then to get number of actions
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512)
            ,nn.ReLU()
            ,nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        pass


class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr, dropout, exploration_max, exploration_min, exploration_decay, double_dq, pretrained):

        self.state_space = state_space
        self.action_space = action_space
        self.double_dq = double_dq
        self.pretrained = pretrained
        self.device = 'cuda' if torn.cuda.is_available() else 'cpu'
        if self.double_dq:
            # algorithm for double deep q network
            self.local_net = DQNSolver(state_space, action_space).to(self.device)
            self.target_net = DQNAgent(state_space, action_space).to(self.device)

            if self.pretrained:
                self.local_net.load_state_dict(torch.load('dq1.pt', map_location=torch.device(self.device)))
                self.target_net.load_state_dict(torch.load('dq2.pt', map_location=torch.device(self.device)))

            self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
            slef.copy = 5000 # updates weights every 5000 steps
            self.step = 0
        else:
            # algorithm for deep q network
            self.dqn = DQNSolver(state_space, action_space).to(self.device)

                if self.pretrained:
                    self.dqn.load_state_dict(torch.load('dq.pt', map_location = torch.device(se;f.device)))
                self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr = lr)


        # creating memory for saves
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load("STATE_MEM.pt")
            self.ACTION_MEM = torch.load("ACTION_MEM.pt")
            self.REWARD_MEM = torch.load("REWARD_MEM.pt")
            self.STATE2_MEM = torch.load("STATE2_MEM.pt")
            self.DONE_MEM = torch.load("DONE_MEM.pt")
            self.ending_position = pickle.load(open("ending_position.pkl", 'rb'))
            self.num_in_queue = pickle.load(open('num_in_queue.pkl', 'rb'))

        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)

        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max # how much we have explored
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember():
        pass

    def recall():
        pass

    def act(self, state):
        '''choose our action for the current state step'''

        if self.double_dq:
            self.step += 1
        if random.random() < self.exploration_rate:
            # if exploration rate is the max, ie new training,\
            # then pull a random action
            return torch.tensor([[random.randrange(self.action_space)]])

        if self.double_dq:
            # find the max probability of the action to take from the local net
            # then reduce dimensionality to return the chosen action
            return torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu
        else:
            # find the max probability of the dqn net
            # then reduce dimensionality to return the chosen action 
            return torch.argmax(self.dqn(state.to(self.device))).unqueeze(0).unsqueeze(0).cpu()
    
    def copy_model():
        pass

    def experience_replay():
        pass



########################################################
# helper functions

def show_state(env, ep=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode = 'rgb_array'))
    plt.title(f'Episode: {ep} {info}')
    plt.axis('off')

    display.clear_output(wait = True)
    display.display(plt.gcf())



########################################################
# main function

def train(training_mode=True, pretrained=False):

    env = make_env('SuperMarioBros-1-1-v0')
    
    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    print('environment is:', env, observation_space, action_space)

    agent = DQNAgent(
        state_space = action_space
        ,max_memory_size = 30000
        ,batch_size = 32
        ,gamma = 0.90
        ,lr = 0.00025
        ,dropout = 0.1 # value was messed
        ,exploration_max = 1.0
        ,exploration_min = 0.02
        ,exploration_decay = 0.99
        ,double_dq = True
        ,pretrained = pretrained
    )

    num_episodes = 100 # training epochs
    env.reset()

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        while True:
            if not training_mode:
                show_state(env, ep_num)
            action = agent.act(state) ########## stopped here
            steps += 1

            state_next, reward, terminal, info = env.step(int(action[0]))

            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)

            terminal = torch.tensor([in(terminal)]).unsqueeze(0)

train()
