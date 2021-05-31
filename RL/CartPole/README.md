# 在倒立摆（CartPole）游戏中实现强化学习
### CartPole简述
Cart Pole即车杆游戏，游戏模型如下图所示。游戏里面有一个小车，上有竖着一根杆子，每次重置后的初始状态会有所不同。小车需要左右移动来保持杆子竖直，为了保证游戏继续进行需要满足以下两个条件：
1. 杆子倾斜的角度θ不能大于15°
2. 小车移动的位置x需保持在一定范围（中间到两边各2.4个单位长度）

![mark](https://external-link.sunan.me/blog/210531/18e8hAa97F.jpg?imageslim)

有两个动作（action）：

左移（0）

右移（1）

四个状态（state）：
1. 小车在轨道上的位置
2. 杆子与竖直方向的夹角
3. 小车速度
4. 角度变化率

### 强化学习模型简述
#### 关键点
对于这个游戏，我们使用了DDQN、Duelling Network以及优先经验回放。

DDQN即Double Q网络，主Q网络和目标Q网络。训练过程是使用目标Q网络训练主Q网络，经过一定的训练步骤后再用主Q网络的参数替换目标Q网络。这样就避免了使用单一Q网络学习过程不稳定的问题。

Duelling Network的示意图和原始Q网络的示意图分别如下图所示(其中A值的平均数为0)：


![mark](https://external-link.sunan.me/blog/210531/m2kfEED5EI.png?imageslim)

![mark](https://external-link.sunan.me/blog/210531/HC4maKH3cg.png?imageslim)

可以看出在Duelling Network中的两个动作都依赖于V,相当于一次更新两个动作的值，无异于提高了学习效率。

优先经验回放是优先学习具有较大TD误差的transition（我们定义的一个具名元组，后有介绍）

TD误差公式：
![mark](https://external-link.sunan.me/blog/210531/1had2JbGkk.png?imageslim)

#### 模型架构
在这个模型中，我们一共创建了五个类，分别是：
1. Environment，环境类，提供游戏的基本环境、游戏玩家（智能体）从该类中或许必要的信息。
2. Agent，智能体类，相当于游戏玩家。
3. Brain，智能体的大脑，决定下一步智能体的动作。该类内部还定义了一个神经网络子类，用来从环境中学习特征输出下一步的动作。
4. ReplayMemory，用于存储经验的内存类。
5. TDErrorMemory，用于存储TD误差的内存类。

此外还定了一个具名元组。
```python
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)
```
### 实现过程
#### 游戏环境（Environment）
该类中有一个初始化方法和一个Run方法。

初始化方法中实例化了智能体和相关参数。

run方法中定义了整个模型的训练过程。第一个循环运行一次代码游戏从开始到结束一次，第二个循环执行一次代表游戏的智能体执行了一个动作，同时会使用经验池中的Batch_size个数据对Q网络的参数进行更新。

当智能体执行了295（自己设定的）个动作之后，游戏仍然没有结束，算作一次成功，奖励加一，否则减一。

当连续10次成功后，则结束训练。

代码如下所示：
```python
class Environment:

    def __init__(self):
        self.env = gym.make(ENV)  # 设置要执行的任务
        num_states = self.env.observation_space.shape[0]  # 设置任务状态和动作的数量
        num_actions = self.env.action_space.n  # CartPole的动作，2
        self.agent = Agent(num_states, num_actions)

    def run(self):
        episode_10_list = np.zeros(10)  # 存储10次实验的连续站立步数，求平均
        complete_episodes = 0  # 持续站立195步，或更多的实验次数
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

                    if step < 195:
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
```

#### 智能体（Agent）
该类有多个方法，其中在初始化方法中实例化了Brain，

详细内容见下述代码和注释。

```python
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
```

#### 智能体的大脑（Brain)
该类是DDQN的核心，拥有多个方法和一个神经网络子类。

在初始化方法中分别实例化了经验存储类、误差存储类、主Q网络和目标Q网络。

reply方法用于更新神经网络参数和Q函数。其中监督值是通过目标Q网络和Q函数得到,再与主Q网络得到的动作值同时进入误差函数，进而反向传播。

reply要执行的任务可以简单分为以下三个：
1. 从经验池中获取小批量数据
2. 通过目标Q网络求得监督信息Q(s,a)
3. 更新神经网络参数

代码如下：

```python
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
```
### 效果演示
![mark](https://external-link.sunan.me/blog/210531/5iJbJaDE5g.gif)
### 源代码
其余代码见GitHub
> https://github.com/mofengboy/Interesting

