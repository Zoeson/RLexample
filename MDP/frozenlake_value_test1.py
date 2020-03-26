#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2020/3/26 10:04
# Author : 
# File : frozenlake_value_test1.py
# Software: PyCharm
# description:

import gym
import numpy as np
from gym import wrappers
from gym.envs.registration import register


def value_iteration(env, gamma):
    """
    v初始值=0，不断根据q_sa进行更新（每个s下，找所有可能的action下，最大的q_sa, 然后v[s]=max q_sa),
    每次迭代都更新一遍v,直到达到停止条件（最大迭代次数、误差很小稳定）
    此时是最优的value, 返回value
    """
    v = np.zeros(env.env.nS)
    max_iteration = 100000
    eps = 1e-20

    # 迭代max_iteration词，
    for i in range(max_iteration):
        prev_v = np.copy(v)

        # 对所有的s进行遍历，
        for s in range(env.env.nS):

            # 找到每个s下最大的q_sa
            q_sa = np.zeros(env.env.nA)
            for a in range(env.env.nA):
                for p, s_, r, _, in env.env.P[s][a]:
                    """
                    s状态下，采取action a之后（即到了另一个状态，
                    比如当前s=位置1，采取action a=0,(在frozenlake里，可能是 动作’上‘），然后有3个可能状态
                    q_sa = Rsa + gamma*sum[所有的s'下：π(s'|s,a) * v(s')]
        env.env.P[s][a]:
                   s=1: {
                     a=0: [  p               s_  r
                        (0.3333333333333333, 1, 0.0, False),
                        (0.3333333333333333, 0, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True)],
                     a=1: [
                        (0.3333333333333333, 0, 0.0, False),
                        (0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 2, 0.0, False)],
                     a=2: [ 
                        (0.3333333333333333, 5, 0.0, True),
                        (0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 1, 0.0, False)],
                     a=3: [
                        (0.3333333333333333, 2, 0.0, False),
                        (0.3333333333333333, 1, 0.0, False),
                        (0.3333333333333333, 0, 0.0, False)]},
                        
                    q_sa[a=0]:
                    比如s == 1, a == 0之后（’上‘）,可能到达的n个状态，即n个可能的s'，在这里用s_表示
                    s_ == 1 时，reward=0, 此处 q(s', a') = p * (r + gamma * prev_v[1]) = p * (0 + gamma* s_的value)
                    s_ == 0 时，reward=0, 此处 q(s', a') = p * (r + gamma * prev_v[0]) = p * (0 + gamma* s_的value)
                    s_ == 5 时，reward=0, 此处 q(s', a') = p * (r + gamma * prev_v[5]) = p * (0 + gamma* s_的value)
                    q_sa[a=1]
                    q_sa[a=2]
                    q_sa[a=3]
                    找出s下，能使q_sa最大的a, 但是我不懂为什么
                    """
                    # s状态下，采取action a之后（即到了另一个状态，
                    # 比如当前s=位置3，采取action a（(0.3333333333333333, 1, 0.0, False),）到了位置3），有3个可能状态
                    q_sa[a] += p * (r + gamma * prev_v[s_])
                    """
                    policy下v[s]的计算
                    q_sa = r + gamma * prev_v[s_]
                    v[s] += p * q_sa
                    """
            v[s] = max(q_sa)
        if np.sum(np.fabs(prev_v - v)) <= eps:
            break
        # print('迭代次数:', i, 'value:', v)
    return v


def extract_policy(v, gamma=1.0):
    """
    这里的，根据v*(上面policy下得到的最优的v), 通过greedy(v*)的方法，得到最优的policy
    与policy_iteration里的extract_policy是完全一样的
    """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        # 每个s下最优的policy
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            for p, s_, r, _ in env.env.P[s][a]:
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)  # greedy(v*)
    return policy


def run_episode(env, policy, gamma=1.0, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma, n=100):
    score = [
            run_episode(env, policy, gamma=gamma, render=False)
            for _ in range(n)]
    return np.mean(score)


if __name__ == '__main__':
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    gamma = 1.0

    optimal_v = value_iteration(env, gamma)
    print(optimal_v)
    policy = extract_policy(optimal_v, gamma)
    print(policy)

    policy_score = evaluate_policy(env, policy, gamma, n=1000)
    print('Policy average score = ', policy_score)



