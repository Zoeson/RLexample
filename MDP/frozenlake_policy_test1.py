#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2020/3/25 14:21
# Author : 
# File : frozenlake_policy_test1.py
# Software: PyCharm
# description:
"""
通过程序，测试并了解 policy_iteration

Q:
    Q1:位置等于状态吗？
    A1:好像是， 从一个位置到另一个位置的概率为p，在程序里的表示好像就是 s-->s_
"""
import gym
from gym import wrappers
from gym.envs.registration import register
import numpy as np


def compute_policy_v(env, policy, gamma=1.0):
    """
    以同样的policy, 每个位置重复n次目前的action（位置s, 重复 policy[s]）
    最终每个位置都得到一个稳定的 value
    返回这个稳定的value
    :param env:
    :param policy:
    :param gamma:
    :return:
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
                v[s] += p * q_sa  # p是在s下采取a的概率? 或者p是从状态s转移到状态s_的概率？！！
        if np.sum((np.fabs(prev_v - v))) <= eps:
            # value converged
            break
    return v


def extract_policy(v, gamma=1.0):
    policy = np.zeros(env.env.nS)  # 16
    for s in range(16):
        q_sa = np.zeros(env.env.nA)  # 4
        for a in range(4):
            for p, s_, r, _ in env.env.P[s][a]:
                q_sa[a] += p * (r + gamma * v[s_])
        policy[s] = np.argmax(q_sa)
    return policy



def policy_iteration(env, gamma=1.0):
    policy = np.random.choice(env.env.nA, size=env.env.nS)
    max_iteration = 200000
    for i in range(max_iteration):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v)
        if np.all(policy == new_policy):
            break
        policy = new_policy
    return policy


def run_episode(env, policy, gamma=1.0, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        # print('int(policy[obs]), obs, reward, done:', int(policy[obs]), obs, reward, done)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    # print(total_reward)
    return total_reward


if __name__ == '__main__':
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    print(env.env.nA, env.env.nS)

    optimal_policy = policy_iteration(env, gamma=1.0)

    t_r = 0
    for _ in range(20):
        result = run_episode(env, optimal_policy)
        print(result)
        t_r += result
    print(t_r/20)




