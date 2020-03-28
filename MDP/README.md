## 0326 结合周博磊的视频，通过程序的实现步骤，增加对policy_iteration和value_iteration的理解
背景：FrozenLake-v0，env.env.nA=4, env.env.nS=16, 4个action， 16个状态

### value_iteration
evalue:用bellman optimal equation直接更新v*,
policy:用greedy(v*)--> policy
```
v = np.zeros(env.env.nS)  
max_iterations = 100000
eps = 1e-20
for i in range(max_iterations):
    prev_v = np.copy(v)
    # 对所有的s进行遍历，
    for s in range(env.env.nS):
        q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)] 
        v[s] = max(q_sa)
    if np.sum(np.fabs(prev_v - v)) <= eps:
        print('Value-iteration converged at iteration# %d.' % (i+1))
        break
        
# ######################################################        
v = np.zeros(env.env.nS)
max_iteration, eps = 100000,1e-20
for i in range(max_iteration):
    prev_v = np.copy(v)
    for s in range(env.env.nS):
        # 找到每个s下最大的q_sa
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            for p, s_, r, _, in env.env.P[s][a]:
                q_sa[a] += p * (r + gamma * prev_v[s_])
                """
                policy下v[s]的计算:
                q_sa = r + gamma * prev_v[s_]
                v[s] += p * q_sa
                """
        v[s] = max(q_sa)
    if np.sum(np.fabs(prev_v - v)) <= eps:
        break
    # print('迭代次数:', i, 'value:', v)
```


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
 
## 0327 进一步理解
### policy_iteration
    扩展：env.env.P,是个什么样的存在？在FrozenLake里，只给了p(s'|s,a), 没有给π(a|s):

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


                    s=0-15分别表示16个位置，a=0-3分别表示：上、下、左、右
                    s状态下，采取action a之后，可能到到另外一些状态，s'<S，
                    比如当前s=位置1，采取action a=0(可能是‘上’），可以 以一定的概率转移到3个可能状态（s_=1,0,5，v[1], v[0],v[5]分别为对应的v(s')），
                    q_sa = R_sa + gamma*sum(所有s'下：p(s'|s,a) * v(s'))  (s_即为 s')

                    q(s1, a0):    q_sa(s=1,a=0):
                        s == 1, a == 0之后（’上‘）,以p的概率，转移到s_的三种状态[1,0,5]

                         R_sa + gamma*sum(所有s'下：p(s'|s,a) * v(s'))
                        = P(s_1|s,a)*r1 + P(s_2|s,a)*r2 + P(s_3|s,a)*r3  +
                            gamma * {P(s_1|s,a)*v(s_1) + P(s_2|s,a)*v(s_2) + P(s_3|s,a)*v(s_3)}
                        = P(s_1|s,a)(r1 + gamma*v(s_1) + P(s_2|s,a)(r2 + gamma*v(s_2) + P(s_3|s,a)(r3 + gamma*v(s_3)

                        即： q_sa = 三个tuple里的 p*(r + gamma*v[s_]) 求和
                                ###
                                s_ == 1 时，reward=0,  p * (0 + gamma * prev_v[1])
                                s_ == 0 时，reward=0,  p * (0 + gamma * prev_v[0])
                                s_ == 5 时，reward=0,  p * (0 + gamma * prev_v[5])
                                ###

                    q(s1, a1):  q_sa[s=1, a=1]
                    q(s1, a2):  q_sa[s=1, a=2]
                    q(s1, a3):  q_sa[s=1, a=3]
                v(s1) = π(a0|s1) * q(s1,a0) + π(a1|s1) * q(s1,a1) + π(a2|s1) * q(s1,a2) + π(a3|s1) * q(s1,a3)
                v(s2) = π(a0|s2) * q(s2,a0) + π(a1|s2) * q(s2,a1) + π(a2|s2) * q(s2,a2) + π(a3|s2) * q(s2,a3)
                v(s3) = π(a0|s3) * q(s3,a0) + π(a1|s3) * q(s3,a1) + π(a2|s3) * q(s3,a2) + π(a3|s3) * q(s3,a3)
                v(s4) = π(a0|s4) * q(s4,a0) + π(a1|s4) * q(s4,a1) + π(a2|s4) * q(s4,a2) + π(a3|s4) * q(s4,a3)
                ...
                ...
                ...
                v(s15) = π(a0|s15)*q(s15,a0) + π(a1|s15)*q(s15,a1) + π(a2|s15)*q(s15,a2) + π(a3|s15)*q(s15,a3)
"""
    
## 问题跟新
### 0326
P(s'|s,a),  π(a|s),  π(a'|s')在FrozenLake_v0这个情景汇总，分别指的是什么
### 0327
    Q1:位置等于状态吗？
    A1:好像是， 从一个位置到另一个位置的概率为p，在程序里的表示好像就是 s-->s_

    Q2:policy到底是什么？
    A2:1. policy是指(当前时刻下\当次迭代时)每个状态对应的，具体要采取的action, for example: FrozenLake-v0 (4*4的棋盘)：
            有（s0, s1, s2, ..., s15）共16个位置（即状态）, 有对应的 [v0, v1, v2, ..., v15],16个位置的state-value
            policy为[a_s0=l, a_s1=r, a_s2=d, a_s3=d, a_s4=u, ..., a_s15=d], 即[l, r, d, d, u, ..., d]
       2. MDP就是为了找 v* (optimal value_state) 以及对应的 π*(a|s): optimal policy, v*是至少对应一个optimal policy
       3. 概念：
           p(s'|s,a): s下，采取a转移到s'的概率？
           π(a|s):   处于环境状态s时，智能体在策略π下采取动作a的概率
           π*(a|s):  s下，最优的a?
           π(s):  从a=π(s)这个式子可以得到，意思是：策略π下，处于环境状态s下，采取的动作。？

    Q3:P_a是dynamic\transmit model for each action?这句话怎么理解
    A3:

    Q4:为什么不断重复当前固定的policy, 各个状态就等得到稳定的v?
    A4:

    Q5：哪些值是确定的？哪些是不定的
    A5: p是确定的，r是确定的， s_是确定的， π(a|s)也是确定的？
        只有v(s_)是不确定的


## Solving FrozenLake using Policy Iteration and Value Iteration

## Introduction
The [FrozenLake env](https://gym.openai.com/envs/FrozenLake-v0/) in OpenAI is a very classic example for Markov Decision Process. Please study the dynamics of the task in detail. Then we will implement the value iteration and policy iteration to search the optimal policy. 

* Load the FrozenLake environment and study the dynamics of the environment:
```
    env_name  = 'FrozenLake-v0' # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    print(env.env.P)
    # it will show 16 states, in which state, there are 4 actions, for each action, there are three possibile states to go with prob=0.333
```

* Run value iteration on FrozenLake
```
python frozenlake_vale_iteration.py
```

* Run policy iteration on FrozenLake
```
python frozenlake_policy_iteration.py
```

* Switch to FrozenLake8x8-v0 for more challenging task.
