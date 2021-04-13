import torch
import torch.nn as nn
import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from tqdm import tqdm
import pickle 
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym
import numpy
import collections 
import cv2
import matplotlib.pyplot as pyplot
from IPython import display
import os


#################################################################################################
## Environment Creation Functions
#################################################################################################

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, gymEnvironment=None, skip=4):
        '''return only every "skip"-th frame'''
        super().__init__(gymEnvironment)

        # Most recent raw observations (for max pooling across time steps)
        self._observation_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        '''pushes the chosen action to the environment'''
        
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            observation, reward, done, info = self.env.step(action) # default rewards from gym
            self._observation_buffer.append(observation) # clears observation deque
            total_reward += reward
            if done:
                break
        max_frame = numpy.max(numpy.stack(self._observation_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        '''clear observation buffer and init to first obs'''
        
        self._observation_buffer.clear()
        observation = self.env.reset()
        self._observation_buffer.append(observation)
        return observation

class ProcessFrame84(gym.ObservationWrapper):
    '''Downsamples image to 84x84, greyscales image, returns numpy array'''
    
    def __init__(self, gymEnvironment=None):
        super().__init__(gymEnvironment)

        self.observation_space = gym.spaces.Box(low=0
                                                ,high=255
                                                ,shape=(84,84,1)
                                                ,dtype = numpy.uint8)

    def observation(self, _observation):
        return ProcessFrame84.process(_observation)
    
    @staticmethod
    def process(frame):
        
        if frame.size == 240 * 256 * 3:
            image = numpy.reshape(frame, [240,256,3]).astype(numpy.float32)
        else:
            assert False, "Unknown resolution."

        image = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
        resized_screen = cv2.resize(image, (84,110), interpolation=cv2.INTER_AREA)
        
        x_t = resized_screen[18:102, :]
        x_t = numpy.reshape(x_t, [84,84,1])
        
        return x_t.astype(numpy.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
    '''makes the image accessible by pytorch'''

    def __init__(self, gymEnvironment):
        super().__init__(gymEnvironment)

        oldShape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0
                                                ,high=1.0
                                                ,shape = (oldShape[-1], oldShape[0], oldShape[1])
                                                ,dtype=numpy.float32)

    def observation(self, observation):
        return numpy.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    '''normalize pixel values in frame -> 0 to 1'''
    
    def observation(self, observation):
        return numpy.array(observation).astype(numpy.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    '''wraps environment space into buffer steps'''
    
    def __init__(self, gymEnvironment, n_steps, dtype = numpy.float32):
        super().__init__(gymEnvironment)
        
        self.dtype = dtype
        old_space = gymEnvironment.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0)
                                                ,old_space.high.repeat(n_steps, axis=0)
                                                ,dtype=dtype)
    
    def reset(self):
        self.buffer = numpy.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())
    
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

#################################################################################
## Making the Environement
#################################################################################

def make_gymEnvironment(gymEnvironment):
    '''takes game version and creates environment from gym_super_mario_bros library'''

    gymEnvironment = MaxAndSkipEnv(gymEnvironment)
    gymEnvironment = ProcessFrame84(gymEnvironment)
    gymEnvironment = ImageToPyTorch(gymEnvironment)
    gymEnvironment = BufferWrapper(gymEnvironment, 4)
    gymEnvironment = ScaledFloatFrame(gymEnvironment)

    return JoypadSpace(gymEnvironment, RIGHT_ONLY)

def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action

    return [0 for _ in range(action)] + [1] + [0 for _ in range(action +1, action_space)]

def show_state(gymEnvironment, epoch = 0, info=""):
    '''shows the current state of mario in his natural environment'''
    
    pyplot.figure(3)
    pyplot.clf()
    pyplot.imshow(gymEnvironment.render(mode = 'rgb_array'))
    pyplot.title(f'Episode: {epoch} {info}')
    pyplot.axis('off')

    display.clear_output(wait = True)
    display.display(pyplot.gcf())


################################################################################################
## Double Deep Q Network Algorithm
################################################################################################

class DQNSolver(nn.Module):

    def __init__(self, input_shape, n_actions):
        super().__init__()
        
        # setting up convolutional network
        # applying ReLU (Rectified Linear Unit) function to nodes and changing node sizes
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
            ,nn.ReLU()
            ,nn.Conv2d(32, 64, kernel_size=4, stride=2)
            ,nn.ReLU()
            ,nn.Conv2d(64, 64, kernel_size=3, stride=1)
            ,nn.ReLU()
        )

        # getting final convolutional network size
        conv_out_size = self._get_conv_out(input_shape)

        # applying sequential linear to 512 (nodes)
        # ReLU, then get number of actions
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512)
                                ,nn.ReLU()
                                ,nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(numpy.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class DQNAgent:
    
    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, learnRate, dropout, exploration_max, exploration_min, exploration_decay, double_dq, pretrained, folder_path):

        # define file space
        self.folder = folder_path
        
        # define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.double_dq = double_dq
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.double_dq:
            # algorithm for double deep q network
            self.local_net = DQNSolver(state_space, action_space).to(self.device)
            self.target_net = DQNSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                self.local_net.load_state_dict(torch.load(f"{self.folder}dq1.pt", map_location=torch.device(self.device)))
                self.target_net.load_state_dict(torch.load(f"{self.folder}dq2.pt", map_location=torch.device(self.device)))
            
            self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=learnRate)
            self.copy = 5000 # copies the local model weights into the target network every 5000 steps
            self.step = 0
        else:
            # deep q network algorithm
            self.dqn = DQNSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                self.dqn.load_state_dict(torch.load("dq.pt", map_location=torch.device(self.device)))
            self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=learnRate)
        
        # creating memory for save states
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load(f"{self.folder}STATE_MEM.pt")
            self.ACTION_MEM = torch.load(f"{self.folder}ACTION_MEM.pt")
            self.REWARD_MEM = torch.load(f"{self.folder}REWARD_MEM.pt")
            self.STATE2_MEM = torch.load(f"{self.folder}STATE2_MEM.pt")
            self.DONE_MEM = torch.load(f"{self.folder}DONE_MEM.pt")
            self.ending_position = pickle.load(open(f"{self.folder}ending_position.pkl", 'rb'))
            self.num_in_queue = pickle.load(open(f"{self.folder}num_in_queue.pkl", 'rb'))
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
        self.l1 = nn.SmoothL1Loss().to(self.device) # also known as huber Loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max # how much we have explored
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
    
    def remember(self, state, action, reward, state2, done):
        '''saves last state to memory'''
        
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size # FIFO Tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
    
    def recall(self):
        '''randomly samples "batch size" memory from the queue'''
        
        idx = random.choices(range(self.num_in_queue)
                             ,k=self.memory_sample_size)

        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]

        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        '''choose our action for the current state step, using epsilon-greedy action'''

        if self.double_dq:
            self.step += 1
        if random.random() < self.exploration_rate:
            # if exploration rate is the max, ie new training,\
            # then pull a random action
            return torch.tensor([[random.randrange(self.action_space)]])
        if self.double_dq:
            # find the max probability of the action to take from the local net
            # then reduce dimensionality to return the chosen action
            return torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
        else:
            # find the max probability of the dqn net
            # then reduce dimensionality to return the chosen action 
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def copy_model(self):
        '''copy local net weights into target net'''

        self.target_net.load_state_dict(self.local_net.state_dict())

    def experience_replay(self):
        
        if self.double_dq and self.step % self.copy == 0:
            self.copy_model()
        
        if self.memory_sample_size > self.num_in_queue:
            return
        
        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()
        if self.double_dq:
            # Double Q-Learning target is Q*(S, A) <- r + y max_a Q_target(S', a)
            target = REWARD + torch.mul((self.gamma * self.target_net(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)

            # local net approximation of Q-value
            current = self.local_net(STATE).gather(1, ACTION.long())
        else:
            # Q-Learning target is Q*(S, A) <- r + y max_a Q(S', a)
            target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)

            current = self.dqn(STATE).gather(1, ACTION.long())
        
        loss = self.l1(current, target)
        loss.backward() # compute gradients
        self.optimizer.step() # backpropagate error

        self.exploration_rate *= self.exploration_decay

        # makes sure that exploration rate is always at least 'exploration min'
        # i.e., adds a hard lower cap to the exploration rate, so that it never reaches 0.
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)


##################################################################################
## MAIN
##################################################################################

def run(training_mode, pretrained, num_episodes, batch_size, gamma):
    ''' main function for running the deep q network '''
    
    fp = f'keeps/e{num_episodes}.bs{batch_size}.gam{gamma}/'
    
    try:
        os.mkdir(fp)
    except:
        pass

    gymEnvironment = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    gymEnvironment = make_gymEnvironment(gymEnvironment)
    observation_space = gymEnvironment.observation_space.shape
    action_space = gymEnvironment.action_space.n
    
    agent = DQNAgent(state_space = observation_space
                      ,action_space = action_space
                      ,max_memory_size = 30000
                      ,batch_size = batch_size
                      ,gamma = gamma
                      ,learnRate = 0.00025
                      ,dropout = 0.1
                      ,exploration_max = 1.0
                      ,exploration_min = 0.02
                      ,exploration_decay = 0.99
                      ,double_dq = True
                      ,pretrained = pretrained
                      ,folder_path = fp)

    
    gymEnvironment.reset()
    total_rewards = []

    for epoch_num in tqdm(range(num_episodes), bar_format='{l_bar}{bar:20}{r_bar}{bar:1b}'):
        state = gymEnvironment.reset()
        state = torch.Tensor([state]) 
        total_reward = 0
        steps = 0
        while True:
            if not training_mode:
                show_state(gymEnvironment, epoch_num)
            action = agent.act(state)
            steps += 1

            state_next, reward, terminal, info = gymEnvironment.step(int(action[0]))
            total_reward += reward   
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)

            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()

            state = state_next
            if terminal:
                break

        total_rewards.append(total_reward)

        print(f" Total reward after episode {epoch_num + 1} is {total_rewards[-1]}")
        num_episodes += 1

    if training_mode:
        pickle.dump(agent.ending_position, open(f"{fp}ending_position.pkl", "wb"))
        pickle.dump(agent.num_in_queue, open(f"{fp}num_in_queue.pkl", "wb"))
        pickle.dump(total_rewards, open(f"{fp}total_rewards.pkl", "wb"))

        if agent.double_dq:
            torch.save(agent.local_net.state_dict(), f"{fp}dq1.pt")
            torch.save(agent.target_net.state_dict(), f"{fp}dq2.pt")
        else:
            torch.save(agent.dqn.state_dict(), f"{fp}dq.pt")
        torch.save(agent.STATE_MEM, f"{fp}STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, f"{fp}ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, f"{fp}REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, f"{fp}STATE2_MEM.pt")
        torch.save(agent.DONE_MEM, f"{fp}DONE_MEM.pt")

    gymEnvironment.close()

    if num_episodes > 500:
        pyplot.title("Episodes trained vs Average Rewards (per 500 EPOCHS)")
        pyplot.plot([0 for _ in range(500)] + numpy.convolve(total_rewards, numpy.ones((500,))/500, mode="valid").tolist())
        pyplot.show()    

if __name__ == '__main__':
    run(True, False, 501, 32, 0.9)