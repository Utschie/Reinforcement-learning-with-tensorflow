from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):#episode就是一个序列的意思，从开始到结束，总共300次过程，每次过程有很多幕，总幕数（包括之前的过程）超200后每5幕学一次
        # initial observation
        observation = env.reset()

        while True:#无限循环直到本次过程终止
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()#从200幕后，每5幕进行一次学习（5就是学习频率learning_freq），更新evaluate-Q网络参数，每200次学习更新一次目标Q网络参数，学习的大脑中的记忆是最近2000幕，也就是可能跨越过程选batch

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1#每进行一幕，step+1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,#每200次学习更新一次target-Q参数
                      memory_size=2000,#记忆容量提到2000，即在最近2000幕中随机选batch
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()