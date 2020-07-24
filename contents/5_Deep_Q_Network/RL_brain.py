"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
#tensorflow 2.0中使用tensorflow 1.0的方法要用tf.compat.v1
import numpy as np
import pandas as pd
import tensorflow as tf


np.random.seed(1)
tf.compat.v1.set_random_seed(1)


# Deep Q Network off-policy，定义了模型的超参数，比如ε-贪心率，折现率γ。
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions#策略空间可选动作数量
        self.n_features = n_features#环境状态特征数
        self.lr = learning_rate#学习率α，也叫步长
        self.gamma = reward_decay#折现率γ
        self.epsilon_max = e_greedy#预设的ε-贪心率最大值，ε是随机选择的概率
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size#批大小，每一次迭代用的样本数量，据说batch size大于1，网络梯度更加稳定
        self.epsilon_increment = e_greedy_increment#给定ε-贪心率参数
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()#建立好网络内的变量关系
        t_params = tf.compat.v1.get_collection('target_net_params')#获取target-Q网络参数
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]#定义target-Q参数更新操作，将来通过run函数来执行这个操作

        self.sess = tf.compat.v1.Session()#将来通过Session().run()来进行传入观测然后进行变量更新

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.compat.v1.train.SummaryWriter soon be deprecated, use following
            tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.cost_his = []#画图用的，就是看每次更新的量应该会越来越小

    def _build_net(self):#其实就是像Matlab一样先定义了各个变量间的关系
        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.n_features], name='s')  # input，Q网络的输入为当前状态s
        self.q_target = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.compat.v1.variable_scope('eval_net'):#在一个名为‘eval_net’的变量空间进行操作
            # c_names(collections_names) are the collections to store variables
            #下面反斜杠后直接回车即可实现续行，但是后面不能跟字符，备注也不行
            #准备下面定义各层时需要的参数
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.compat.v1.random_normal_initializer(0., 0.3), tf.compat.v1.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.compat.v1.variable_scope('l1'):#
                #get_variahble():如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的(tf1.0方法)
                #设置第一层的w1和b1，w1为n_features*n_l1的矩阵，用initializer来初始化
                #把w1，b1分别加入‘eval_net_params’(好像刚才就自动创建了)和GLOBAL_VARIABLES的集合
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(self.s, w1) + b1)#括号里是线性计算公式，nn.relu是用relu作非线性激活函数，l1则是第一层输出

            # second layer. collections is used later when assign to target net
            with tf.compat.v1.variable_scope('l2'):#建立第二层的权重张量和偏置
                w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.compat.v1.matmul(l1, w2) + b2#得到第二层输出，是一个跟动作空间一样长的向量

        with tf.compat.v1.variable_scope('loss'):#定义算是损失函数变量
            self.loss = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(self.q_target, self.q_eval))#用q_target和q_eval定义损失函数
        with tf.compat.v1.variable_scope('train'):#下面的RMSProp是梯度下降法的一种
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)#用run执行了_train_op后权重自动更新

        # ------------------ build target_net ------------------定义target-Q网络，该网络的输入为下一状态s_
        self.s_ = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.n_features], name='s_')    # input
        with tf.compat.v1.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.compat.v1.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):#把每一幕的状态和动作二元组传给memory，每500幕分配一个index，作为一组
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0#幕数

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition#index最大值也就499，也就是一个memory最多能存储500次转移，如果又来了一轮，则用最新的转移代替最老的，保证memory里总是最新的500次转移
                   #此处的引用方式是numpy对象的引用方式，切片啊啥的，memory是一个高维数组，第一个维度是index，各个维度下面是一个元组
        self.memory_counter += 1#进行转移的幕数

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})#observation即为每一幕的观测状态s的输入，通过Session().run()函数来给q_eval赋值，并将q_eval返回
            action = np.argmax(actions_value)#用np.argmax得到q_eval中最大值的索引，而索引与动作一一对应
        else:
            action = np.random.randint(0, self.n_actions)
        return action#把动作的索引传出去

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:#每更新300次evaluate-Q网络参数后，更新一次target-Q网络参数
            self.sess.run(self.replace_target_op)#
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:#如果进行的总幕数多于500
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)#在最近500次转移中随机选出32次转移
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]#把随机选出来的最近500次中的32次转移作为batch

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })#每一次学习都用随机选择的batch_size数量的幕数求出q_next, q_eval

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)#求出新的q_target值

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})#用刚才的q_target值把网络中的q_target更新，求出loss，再运行train_op更新evaluate-Q网络参数
        self.cost_his.append(self.cost)

        # increasing epsilon，提高随机率
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1#evaluate网络参数每更新一次(learn函数每运行一次)，学习步数+1

    def plot_cost(self):#画图
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()





