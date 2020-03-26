## 0326 结合周博磊的视频，通过程序的实现步骤，增加对policy_iteration和value_iteration的理解
背景：FrozenLake-v0，env.env.nA=4, env.env.nS=16, 4个action， 16个状态

### policy_iteration

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
    
    
### 我的问题：

P(s'|s,a),  π(a|s),  π(a'|s')在FrozenLake_v0这个情景汇总，分别指的是什么
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
