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
        old_source = env.observation_space
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
# main function

def train(training_mode=True, pretrained=False):

    env = make_env('SuperMarioBros-1-1-v0')
    
    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    print('environment is:', env, observation_space, action_space)

    # agent = stuff

train()
