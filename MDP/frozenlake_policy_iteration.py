"""
Solving FrozenLake environment using Policy-Iteration.

Adapted by Bolei Zhou for IERG6130. Originally from Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """
    以同样的policy, 每个位置重复n次目前的action（位置s, 重复 policy[s]）
    最终每个位置都得到一个稳定的 value
    返回这个稳定的value
    """
    v = np.zeros(env.env.nS)
    eps = 1e-10
    i = 0  # 迭代次数
    while True:
        i += 1
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            # todo s:0-15
            # todo: a:policy_a 就是现阶段的策略，状态s下采取的action
            policy_a = policy[s]
            v[s] = 0
            for p, s_, r, _ in env.env.P[s][policy_a]:  # 在s和a下，求q
                # #####print(i, s, policy_a, p, s_, r)  # p,s_,r:0.333, 4, 0.0
                # r:reword, 只有在s_是15的时候，才有+1的reword
                # s_: 下一个状态, 下一个状态的value:prev_v[s_]
                q_sa = r + gamma * prev_v[s_]
                # v_π_s = 所有a属于A, π(a|s) * q(s,a)
                # Q:p是在s下采取a的概率? 或者p是从状态s转移到状态s_的概率？
                v[s] += p * q_sa  
                # A:在这个里面，action,就是表示从一个位置（state）到另一个位置（state）, 所有，π(a|s)既是s下采取a行动的概率，也即s下去到另一个state的概率
        if np.sum((np.fabs(prev_v - v))) <= eps:
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.env.nA, size=(env.env.nS))  # initialize a random policy
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

if __name__ == '__main__':

    env_name  = 'FrozenLake-v0' # 'FrozenLake8x8-v0'
    env = gym.make(env_name)

    optimal_policy = policy_iteration(env, gamma = 1.0)
    scores = evaluate_policy(env, optimal_policy, gamma = 1.0)
    print('Average scores = ', np.mean(scores))
    
   
"""
s_ == 15时，r == 1,
s_ == 其它，r == 0
s_、s 在frozenLake里表示位置，相当于课程中的state状态， prev_v[s_]是当前位置的value值， 通过不断更新v, prev_v=np.copy(v)不断更新prev_v
'P': {
0: {
   0: [
     p,                  s_, r,    _,
    (0.3333333333333333, 0, 0.0, False),
    (0.3333333333333333, 0, 0.0, False),
    (0.3333333333333333, 4, 0.0, False)],
   1: [
    (0.3333333333333333, 0, 0.0, False),
    (0.3333333333333333, 4, 0.0, False),
    (0.3333333333333333, 1, 0.0, False)],
   2: [
    (0.3333333333333333, 4, 0.0, False),
    (0.3333333333333333, 1, 0.0, False),
    (0.3333333333333333, 0, 0.0, False)],
   3: [
    (0.3333333333333333, 1, 0.0, False),
    (0.3333333333333333, 0, 0.0, False),
    (0.3333333333333333, 0, 0.0, False)]},
1: {
   0: [
    (0.3333333333333333, 1, 0.0, False),
    (0.3333333333333333, 0, 0.0, False),
    (0.3333333333333333, 5, 0.0, True)],
   1: [
    (0.3333333333333333, 0, 0.0, False),
    (0.3333333333333333, 5, 0.0, True),
    (0.3333333333333333, 2, 0.0, False)],
   2: [ 
    (0.3333333333333333, 5, 0.0, True),
    (0.3333333333333333, 2, 0.0, False),
    (0.3333333333333333, 1, 0.0, False)],
   3: [
    (0.3333333333333333, 2, 0.0, False),
    (0.3333333333333333, 1, 0.0, False),
    (0.3333333333333333, 0, 0.0, False)]},
2: {
   0: [
    (0.3333333333333333, 2, 0.0, False),
    (0.3333333333333333, 1, 0.0, False),
    (0.3333333333333333, 6, 0.0, False)],
   1: [
    (0.3333333333333333, 1, 0.0, False),
    (0.3333333333333333, 6, 0.0, False),
    (0.3333333333333333, 3, 0.0, False)],
   2: [ 
    (0.3333333333333333, 6, 0.0, False),
    (0.3333333333333333, 3, 0.0, False),
    (0.3333333333333333, 2, 0.0, False)],
   3: [
    (0.3333333333333333, 3, 0.0, False),
    (0.3333333333333333, 2, 0.0, False),
    (0.3333333333333333, 1, 0.0, False)]},
3: {
   0: [
    (0.3333333333333333, 3, 0.0, False),
    (0.3333333333333333, 2, 0.0, False),
    (0.3333333333333333, 7, 0.0, True)],
   1: [ 
    (0.3333333333333333, 2, 0.0, False),
    (0.3333333333333333, 7, 0.0, True),
    (0.3333333333333333, 3, 0.0, False)],
   2: [
    (0.3333333333333333, 7, 0.0, True),
    (0.3333333333333333, 3, 0.0, False),
    (0.3333333333333333, 3, 0.0, False)],
   3: [
    (0.3333333333333333, 3, 0.0, False),
    (0.3333333333333333, 3, 0.0, False),
    (0.3333333333333333, 2, 0.0, False)]},
 4: {
   0: [
    (0.3333333333333333, 0, 0.0, False),
    (0.3333333333333333, 4, 0.0, False),
    (0.3333333333333333, 8, 0.0, False)],
   1: [
    (0.3333333333333333, 4, 0.0, False),
    (0.3333333333333333, 8, 0.0, False),
    (0.3333333333333333, 5, 0.0, True)],
   2: [
    (0.3333333333333333, 8, 0.0, False),
    (0.3333333333333333, 5, 0.0, True),
    (0.3333333333333333, 0, 0.0, False)],
   3: [
    (0.3333333333333333, 5, 0.0, True),
    (0.3333333333333333, 0, 0.0, False),
    (0.3333333333333333, 4, 0.0, False)]},
    .
    .
    .

13: {
   0: [
    (0.3333333333333333, 9, 0.0, False),
    (0.3333333333333333, 12, 0.0, True),
    (0.3333333333333333, 13, 0.0, False)],
   1: [
    (0.3333333333333333, 12, 0.0, True),
    (0.3333333333333333, 13, 0.0, False),
    (0.3333333333333333, 14, 0.0, False)],
   2: [
    (0.3333333333333333, 13, 0.0, False),
    (0.3333333333333333, 14, 0.0, False),
    (0.3333333333333333, 9, 0.0, False)],
   3: [
    (0.3333333333333333, 14, 0.0, False),
    (0.3333333333333333, 9, 0.0, False),
    (0.3333333333333333, 12, 0.0, True)]},
14: {
   0: [
    (0.3333333333333333, 10, 0.0, False),
    (0.3333333333333333, 13, 0.0, False),
    (0.3333333333333333, 14, 0.0, False)],
   1: [ 
    (0.3333333333333333, 13, 0.0, False),
    (0.3333333333333333, 14, 0.0, False),
    (0.3333333333333333, 15, 1.0, True)],
   2: [
    (0.3333333333333333, 14, 0.0, False),
    (0.3333333333333333, 15, 1.0, True),
    (0.3333333333333333, 10, 0.0, False)],
   3: [
    (0.3333333333333333, 15, 1.0, True),
    (0.3333333333333333, 10, 0.0, False),
    (0.3333333333333333, 13, 0.0, False)]},
15: {
   0: [(1.0, 15, 0, True)],
   1: [(1.0, 15, 0, True)],
   2: [(1.0, 15, 0, True)],
   3: [(1.0, 15, 0, True)]}},
"""
