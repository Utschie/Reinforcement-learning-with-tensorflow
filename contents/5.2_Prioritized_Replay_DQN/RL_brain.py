"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0#数据指针，作为存储数据的那个向量self.data的位置指针,初始化为0，即从第一个空位存起

    def __init__(self, capacity):#树的capacity即为回放区的大小，即放在回放区样本的个数
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)#树由一个一维向量组成，每个元素存储一个节点的p，每个外部节点（叶节点）对应一次转移的p，初始化所有的p=0
        #二叉树性质：对于一棵满二叉树，如果外部节点（叶节点）的个数为n，则内部节点的个数为n-1
        #所以节点个数如下：                                                                         
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        #是一个一维向量，用来存储每一次转移，所以每个元素是一次转移，数据类型为对象，所以大小=回放区容量
        #实际上self.data和后self.capacity个叶节点一一对应
        # [--------------data frame-------------]
        #             size: capacity
    
    # 当有新 sample 时, 添加进 tree 和 data
    def add(self, p, data):#给出一次转移和其对应的p，向树上增加节点
        tree_idx = self.data_pointer + self.capacity - 1#tree_idx是该次转移的p的存储的位置，之所以加上self.capacity-1是因为要跳过前面的self.capacity-1个父节点
        self.data[self.data_pointer] = data  # update data_frame，在指针对应的位置放入data
        self.update(tree_idx, p)  # update tree_frame，把data所对应的叶节点的p更新，同时更新与之相关联向上的所有父节点的p

        self.data_pointer += 1#数据指针指向下一个位置
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity，如果指针指到self.data的末尾了，即tree_idx到了self.tree的末尾，即回放区装满了，则重新开始
            self.data_pointer = 0
    # 当 sample 被 train, 有了新的 TD-error, 就在 tree 中更新
    def update(self, tree_idx, p):#给出叶节点的位置和p，更新树
        change = p - self.tree[tree_idx]#得到新p和旧p的差
        self.tree[tree_idx] = p#给新p赋给那个叶节点
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            #由于叶节点的p变化了，于是响应的其所有相关联的父节点的p都要更新，直到更新到根节点（tree_idx = 0）
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):#单个样本抽取（采样）过程
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0#从根节点开始找起
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1 #该父节点左边的子节点对应在tree向量里的位置        # this leaf's left and right kids，由于是用一维向量存储的数，所以各个点的相对位置由这样的公式确定
            cr_idx = cl_idx + 1#该父节点右边的子节点对应在tree向量里的位置
            if cl_idx >= len(self.tree):        # reach bottom, end search
                #如果计算出的左节点位置超出了树的长度，即parent_idx >= capacity - 1（父节点总共n-1个，则最大坐标为n-2），即父节点已经是叶节点了
                leaf_idx = parent_idx#则此时就已经到达那个叶节点了
                break
            else:       # downward search, always search for a higher priority node,如果没到底，则继续向下搜索
                if v <= self.tree[cl_idx]:#如果v小于等于左子节点的p
                    parent_idx = cl_idx#则在左子节点向下搜索
                else:#如果v大于左子节点的p
                    v -= self.tree[cl_idx]#则v减去左子节点的p更新为新的v
                    parent_idx = cr_idx#并以右子节点为父节点

        data_idx = leaf_idx - self.capacity + 1#该叶节点在数中的位置对应转换成在data里的位置
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]#返回这个样本的叶节点坐标，p和转移数据

    @property#python内置装饰器，把方法作为属性调用，即可以直接调用total_p获得根节点的p值
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree，一个记忆回放区的类，里面就是一棵书，以及和环境交互的抽样和p值计算方法
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):#记忆回放区就是一棵树，存储着记忆数据和其对应的p以及整个树上的p，(p/total_p)即为某个样本被抽中的概率
        self.tree = SumTree(capacity)

    def store(self, transition):#把一次转移交给memory并给它赋予p值然后存储起来
        max_p = np.max(self.tree.tree[-self.tree.capacity:])#在树的所有叶节点（即从后数capacity个及其后面所有的节点）上找到最大的p值
        if max_p == 0:#如果最大叶节点的p值为0，即树是空的
            max_p = self.abs_err_upper#则把最大的p值定为1.0
        self.tree.add(max_p, transition)#以已存在的最大p为p，在树中增加该次转移   # set the max p for new p，
    #抽取每次训练的batch
    def sample(self, n):#每次抽样的过程，其中n为batch_size,即抽出的样本个数
        #b_idx存储的是抽取的batch在tree中的位置index，(n,)是形状参数，即创建一个一维的有n个元素的数组，其中元素由于empty机制是随机数而不是空
        #b_memory存储的是抽取的batch的数据，形状参数为(batch_size,data的长度)，即一个【batch_size*每次转移存储进去的指标数(比如state,capital,next_state....)】形式的矩阵
        #ISWeights是Importance Sampling Weights，即重要度抽样权重，是一个batch_size*1的二维数组，每个元素是一个长度为1的一维数组
        #ISWeights是用于修改损失函数，需要考虑权重，即拥有更大权重的样本的TD-error对总loss贡献更大
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n#把p按着总p平均分成batch_size份       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p #所有样本中最小被抽取的概率，即所有的p/total_p中的最小值    # for later calculate ISweight
        for i in range(n):#在每个被用p分出的份儿中随机选取一个v
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)#在每个被用p分出的份儿中随机选取一个v
            idx, p, data = self.tree.get_leaf(v)#根据v获得被抽出来的样本的位置，p和数据值
            prob = p / self.tree.total_p#计算该样本被抽出来的概率
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)#np.power是求幂函数，每个样本的ISweights = (其被抽概率/整个树中最小被抽取概率)^beta
            b_idx[i], b_memory[i, :] = idx, data#把这个样本的位置，数据值作为batch里对应的元素
        return b_idx, b_memory, ISWeights
    # train 完被抽取的 samples 后更新在 tree 中的 sample 的 priority
    def batch_update(self, tree_idx, abs_errors):#这里的tree_idx和abs_errors分别是一个有batch_size个元素的列表
        abs_errors += self.epsilon  # convert to abs and avoid 0，纯粹是为了避免td-error（绝对值误差）为0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)#限制td-error，如果td-error大于1，则使其最大为1
        ps = np.power(clipped_errors, self.alpha)#根据训练后的td-error和alpha计算出batch中各样本新的p
        for ti, p in zip(tree_idx, ps):#按着样本在书中的位置
            self.tree.update(ti, p)#更新对应的整个树的p


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,  trainable=trainable)
                out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, True)

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)

    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
