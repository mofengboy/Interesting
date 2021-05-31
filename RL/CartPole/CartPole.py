import numpy as np
import matplotlib.pyplot as plt
import gym
from matplotlib import animation
from collections import namedtuple
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# 常量
ENV = 'CartPole-v1'  # 任务名
GAMMA = 0.99  # 时间折扣率
MAX_STEPS = 500  # 1次试验中的step数
NUM_EPISODES = 1000  # 最大尝试次数
BATCH_SIZE = 32
CAPACITY = 10000


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                   interval=50)

    anim.save('movie_cartpole_DQN.mp4')
    plt.show()


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)

TD_ERROR_EPSILON = 0.0001


# TD误差存储类
class TDErrorMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, td_error):

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def get_prioritized_indexes(self, batch_size):
        '''根据TD误差以概率获得index'''

        # 计算TD误差总和
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 添加一个微小值

        # 为batch_size 生成随机数并且按照升序排列
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (
                        abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1

            # 使用微小值进行计算超出内存
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes

    def update_td_error(self, updated_td_errors):
        self.memory = updated_td_errors


# 用于存储经验的内存类
class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # memory的最大长度
        self.memory = []  # 存储过往经验
        self.index = 0  # 表示要保存的数据

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 内存未满时添加

        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        # 随机检索batch_size大小的样本返回
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # 返回当前memory的长度
        return len(self.memory)


class Brain:
    def __init__(self, num_states, num_actions):
        # 获取CartPole的两个动作（向左或向右）
        self.num_actions = num_actions

        # 创建存储经验的对象
        self.memory = ReplayMemory(CAPACITY)

        # 主Q网络
        self.main_q_network = self.Net(num_states, num_actions)
        # 目标Q网络
        self.target_q_network = self.Net(num_states, num_actions)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

        self.td_error_memory = TDErrorMemory(CAPACITY)

    def replay(self, episode):
        '''通过经验回放学习参数'''

        # 若经验池大小小于小批量数据时，不执行任何操作
        if len(self.memory) < BATCH_SIZE:
            return

        #  创建小批量数据
        self.batch, self.state_batch, self.action_batch, self.reward_batch, \
        self.non_final_next_states = self.make_mini_batch(episode)

        # 求Q(s_t,a_t)作为监督信息
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 更新参数
        self.update_main_q_network()

    class Net(nn.Module):
        def __init__(self, num_states, num_actions):
            super().__init__()
            self.fc1 = nn.Linear(num_states, 32)
            self.fc2 = nn.Linear(32, 32)
            # dueling network
            self.fc3_adv = nn.Linear(32, num_actions)  # Advantage 部分
            self.fc3_v = nn.Linear(32, 1)  # 价值V部分

        def forward(self, x):
            y = F.relu(self.fc1(x))
            y = F.relu(self.fc2(y))

            adv = self.fc3_adv(y)
            val = self.fc3_v(y).expand(-1, adv.size(1))

            output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
            return output

    def make_mini_batch(self, episode):
        # 从经验池中获取小批量数据
        if episode < 30:
            transitions = self.memory.sample(BATCH_SIZE)
        else:
            indexes = self.td_error_memory.get_prioritized_indexes(BATCH_SIZE)
            transitions = [self.memory.memory[n] for n in indexes]

        # 将(state, action, state_next, reward)×BATCH_SIZE
        # 转换为(state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 将每个变量的元素转为与小批量数据对应的形式
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        # 找到Q(s_t,a_t）作为监督信息

        self.main_q_network.eval()
        self.target_q_network.eval()

        # 网络输出的Q(s_t,a_t)
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        # 求max{Q(s_t+1, a)}的值
        # 创建索引掩码以检查CartPole是否完成且具有next_state
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        # 首先全部设置为0
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).long()
        # 从主Q网络求下一个状态中最大Q值的动作a_m，并返回该动作对应的索引
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]

        # 仅过滤有下一状态的，并将size32转换成size 32*1
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 从目标Q网络求出具有下一状态的index的动作a_m的最大Q值
        # 使用squeeze（）将size[minibatch*1]压缩为[minibatch]
        next_state_values[non_final_mask] = \
            self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states)\
                .detach().squeeze()

        # 从Q公式中求Q(s_t, a_t)作为监督信息
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def update_td_error_memory(self):

        self.main_q_network.eval()
        self.target_q_network.eval()

        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_action_values = self.main_q_network(
            state_batch).gather(1, action_batch)

        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        next_state_values = torch.zeros(len(self.memory))
        a_m = torch.zeros(len(self.memory)).long()

        a_m[non_final_mask] = self.main_q_network(non_final_next_states).detach().max(1)[1]

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(
            non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 找到TD误差
        td_errors = (reward_batch + GAMMA * next_state_values) - \
                    state_action_values.squeeze()

        self.td_error_memory.memory = td_errors.detach().numpy().tolist()

    def decide_action(self, state, episode):
        '''根据当前状态确定动作'''
        # 采用ε-贪婪发逐步采用最佳动作
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
                # [torch.LongTensor of size 1]
        else:
            # 随机返回0、1动作
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        return action


class Agent:
    # 智能体，带杆的小车

    def __init__(self, num_states, num_actions):
        '''设置任务状态和动作的数量'''
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self, episode):
        '''更新Q函数'''
        self.brain.replay(episode)

    def get_action(self, state, episode):
        '''确定动作'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''将state, action, state_next, reward保存在经验池中'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        # 将目标Q网络更新到与主网络相同
        self.brain.update_target_q_network()

    def memorize_td_error(self, td_error):
        # 存储TD误差
        self.brain.td_error_memory.push(td_error)

    def update_td_error_memory(self):
        # 更新存储在TD误差存储器中的误差
        self.brain.update_td_error_memory()


class Environment:

    def __init__(self):
        self.env = gym.make(ENV)  # 设置要执行的任务
        num_states = self.env.observation_space.shape[0]  # 设置任务状态和动作的数量
        num_actions = self.env.action_space.n  # CartPole的动作，2
        self.agent = Agent(num_states, num_actions)

    def run(self):
        episode_10_list = np.zeros(10)  # 存储10次实验的连续站立步数，求平均
        complete_episodes = 0  # 持续站立295步，或更多的实验次数
        episode_final = False  # 最终尝试标志
        frames = []  # 用于存储图像的变量，使最后一轮成为动画

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            state = observation
            state = torch.from_numpy(state).float()
            state = torch.unsqueeze(state, 0)  # 将size 4转为size 1x4

            for step in range(MAX_STEPS):

                if episode_final is True:
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(state, episode)

                # 通过动作a_t得到s_{t+1}和done标志
                observation_next, _, done, _ = self.env.step(action.item())

                if done:  # step超过200，或者倾斜超过某个角度，done为True
                    state_next = None  # 没有下一个状态，存储为None

                    # 添加最近10轮的站立步数到列表中
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))

                    if step < 295:
                        reward = torch.from_numpy(np.array([-1.0])).float()  # 中途倒下，奖励-1作为惩罚
                        complete_episodes = 0  # 重置连续成功几率
                    else:
                        reward = torch.from_numpy(np.array([1.0])).float()  # 一直站立到结束时，奖励1
                        complete_episodes = complete_episodes + 1
                else:
                    reward = torch.from_numpy(np.array([0.0])).float()  # 正常奖励为0
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).float()
                    state_next = torch.unsqueeze(state_next, 0)

                # 向经验池中添加经验
                self.agent.memorize(state, action, state_next, reward)

                # 将TD误差添加到误差变量中
                self.agent.memorize_td_error(0)

                # 优先经验回放中更新Q函数
                self.agent.update_q_function(episode)

                state = state_next

                if done:
                    print('%d Episode: Finished after %d steps：10次试验平均的step数 = %.1lf' % (
                        episode, step + 1, episode_10_list.mean()))

                    # 更新TD误差存储变量中的内容
                    self.agent.update_td_error_memory()

                    if episode % 2 == 0:
                        self.agent.update_target_q_function()
                    break

            if episode_final is True:
                # 保存并绘制动画
                display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:
                print('连续10轮成功')
                episode_final = True  # 使下一次尝试成为最终的动画


if __name__ == '__main__':
    cartpole_env = Environment()
    cartpole_env.run()
