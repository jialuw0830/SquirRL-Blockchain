"""
Optimized eth.py (TF1) - lower GPU memory + faster training

Key changes:
1) TF Session allow_growth (no pre-alloc full GPU mem)
2) targetQN does NOT build optimizer/Adam slots (saves memory)
3) soft target update via tf.group (single sess.run)
4) Numpy ring replay buffer (faster + less memory fragmentation)
5) reduce sess.run round-trips in training step
6) gather_nd instead of one-hot for Q(s,a)
"""

from __future__ import division

import gym
import numpy as np
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import time

from environment import SM_env
from environment import eth_env

# -----------------------------
# Replay Buffer (fast, numpy ring)
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity,), dtype=np.int32)
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)  # 0/1
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s1, d):
        i = self.ptr
        self.state[i] = s
        self.next_state[i] = s1
        self.action[i] = int(a)
        self.reward[i] = float(r)
        self.done[i] = 1.0 if d else 0.0
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        bs = int(batch_size)
        idx = np.random.randint(0, self.size, size=bs)
        return (self.state[idx],
                self.action[idx],
                self.reward[idx],
                self.next_state[idx],
                self.done[idx])


# -----------------------------
# Q Network (Dueling DQN)
# trainable=False => no optimizer/adam slots created
# -----------------------------
class Qnetwork:
    def __init__(self, h_size, state_space_n, state_vector_n, action_space_n,
                 scope="qnet", trainable=True):

        self.state_space_n = state_space_n
        self.action_space_n = action_space_n
        self.state_vector_n = state_vector_n

        with tf.variable_scope(scope):
            self.vectorIn = tf.placeholder(shape=[None, state_vector_n], dtype=tf.float32, name="state")

            fc1 = tf.layers.dense(self.vectorIn, h_size, activation=tf.nn.relu, name="fc1")
            fc2 = tf.layers.dense(fc1, h_size, activation=tf.nn.relu, name="fc2")

            streamA, streamV = tf.split(fc2, 2, axis=1)

            init = tf.glorot_uniform_initializer()
            AW = tf.get_variable("AW", shape=[h_size // 2, action_space_n], initializer=init)
            VW = tf.get_variable("VW", shape=[h_size // 2, 1], initializer=init)

            Advantage = tf.matmul(streamA, AW)
            Value = tf.matmul(streamV, VW)

            self.Qout = Value + (Advantage - tf.reduce_mean(Advantage, axis=1, keepdims=True))
            self.predict = tf.argmax(self.Qout, axis=1, output_type=tf.int32)

            if trainable:
                self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32, name="targetQ")
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

                # Q(s,a) via gather_nd (faster + less memory than one-hot)
                batch_idx = tf.range(tf.shape(self.actions)[0], dtype=tf.int32)
                indices = tf.stack([batch_idx, self.actions], axis=1)
                Q_sa = tf.gather_nd(self.Qout, indices)

                td_error = tf.square(self.targetQ - Q_sa)
                self.loss = tf.reduce_mean(td_error)

                opt = tf.train.AdamOptimizer(learning_rate=1e-4)
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                self.updateModel = opt.minimize(self.loss, var_list=var_list)

    def act_epsilon_greedy(self, sess, s_vec, e=0.0):
        # s_vec: np.float32 vector for NN input
        if np.random.rand() < e:
            return np.random.randint(self.action_space_n)
        # fetch only argmax action (less device->host transfer than full Q vector)
        a = sess.run(self.predict, feed_dict={self.vectorIn: s_vec[None, :]})
        return int(a[0])


# -----------------------------
# Target network update ops (single sess.run)
# -----------------------------
def make_soft_update_ops(main_scope="main", target_scope="target", tau=0.001):
    main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope)
    main_vars = sorted(main_vars, key=lambda v: v.name)
    target_vars = sorted(target_vars, key=lambda v: v.name)

    ops = []
    for vm, vt in zip(main_vars, target_vars):
        ops.append(vt.assign(tau * vm + (1.0 - tau) * vt))
    return tf.group(*ops)


# -----------------------------
# Hyperparams (keep yours)
# -----------------------------
batch_size = 32
update_freq = 4
y = 0.99
startE = 1.0
endE = 0.1
annealing_steps = 10000.0
num_episodes = 100
pre_train_steps = 10000
max_epLength = 10000
load_model = False
load_best_model = False
h_size = 80
tau = 0.001

know_alpha = True
ALPHA = 0.4
GAMMA = 0.5
HIDDEN_BLOCK = 20
DEV = 0.0
interval = (0, 0.5)

path = "./eth_" + str(HIDDEN_BLOCK) + "_" + str(h_size)
if know_alpha:
    path += "know_alpha"
best_path = path + "/model_best.ckpt"
file = open("eth_" + str(HIDDEN_BLOCK) + "_" + str(h_size) + ".txt", "a+")
file.write("\n\n\n\n A new test from now!")

env = eth_env(
    max_hidden_block=HIDDEN_BLOCK,
    attacker_fraction=ALPHA,
    follower_fraction=GAMMA,
    dev=DEV,
    random_interval=interval,
    know_alpha=know_alpha,
    relative_p=0.54
)

# -----------------------------
# Build graph
# -----------------------------
tf.reset_default_graph()
mainQN = Qnetwork(h_size, env._state_space_n, env._state_vector_n, env._action_space_n,
                  scope="main", trainable=True)
targetQN = Qnetwork(h_size, env._state_space_n, env._state_vector_n, env._action_space_n,
                    scope="target", trainable=False)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=3)

hard_update_op = make_soft_update_ops("main", "target", tau=1.0)   # init copy
soft_update_op = make_soft_update_ops("main", "target", tau=tau)   # EMA update

# Replay buffer
myBuffer = ReplayBuffer(capacity=50000, state_dim=env._state_vector_n)

# epsilon schedule
e = startE
stepDrop = (startE - endE) / annealing_steps

rList = []
fList = []
jList = []
total_steps = 0
history_best = 0.0

if not os.path.exists(path):
    os.makedirs(path)

# -----------------------------
# TF session config: reduce GPU mem usage
# -----------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
    sess.run(init)
    sess.run(hard_update_op)  # important: target starts identical to main

    if load_model:
        print("Loading Model...")
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    if load_best_model:
        print("Loading Best Model..")
        saver.restore(sess, best_path)

    train_start = time.time()

    for ep in range(num_episodes):
        ep_start = time.time()

        s_raw = env.reset()
        s_vec = np.asarray(s_raw, dtype=np.float32)

        rAll = 0.0
        for j in range(1, max_epLength + 1):
            # action
            if total_steps < pre_train_steps:
                a = np.random.randint(env._action_space_n)
            else:
                a = mainQN.act_epsilon_greedy(sess, s_vec, e)

            # step env (use raw state for env)
            s1_raw, r, d, _ = env.step(s_raw, int(a), move=True)
            s1_vec = np.asarray(s1_raw, dtype=np.float32)

            # store experience (vector form)
            myBuffer.add(s_vec, a, r, s1_vec, d)

            total_steps += 1
            rAll += float(r)

            # train
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if (total_steps % update_freq == 0) and (myBuffer.size >= batch_size):
                    states, actions, rewards, next_states, dones = myBuffer.sample(batch_size)

                    # Double-DQN target:
                    # a* = argmax_a Q_main(s',a); target uses Q_target(s',a*)
                    q_main_next, q_target_next = sess.run(
                        [mainQN.Qout, targetQN.Qout],
                        feed_dict={mainQN.vectorIn: next_states, targetQN.vectorIn: next_states}
                    )
                    a_next = np.argmax(q_main_next, axis=1).astype(np.int32)
                    doubleQ = q_target_next[np.arange(batch_size), a_next]
                    targetQ = rewards + y * doubleQ * (1.0 - dones)

                    # one sess.run for train + target update
                    sess.run(
                        [mainQN.updateModel, soft_update_op],
                        feed_dict={mainQN.vectorIn: states,
                                   mainQN.actions: actions,
                                   mainQN.targetQ: targetQ}
                    )

            s_raw, s_vec = s1_raw, s1_vec
            if d:
                break

        # logging
        rList.append(rAll)
        fList.append(env.reward_fraction)
        jList.append(j)

        print("round = ", ep, "training steps = ", j, "reward = ", rAll,
              "frac = ", env.reward_fraction, "elapsed_sec = ", time.time() - ep_start)
        print("round = ", ep, "training steps = ", j, "reward = ", rAll,
              "frac = ", env.reward_fraction, file=file)

        # save best
        if env.reward_fraction > history_best:
            history_best = env.reward_fraction
            saver.save(sess, best_path)

        # periodic prints
        if (ep + 1) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]), e)

        # (optional) periodic checkpoint (your old i%1000 was basically never triggered for 100 eps)
        if (ep + 1) % 100 == 0:
            saver.save(sess, path + "/model-" + str(ep + 1) + ".ckpt")

        # NOTE: 你原来每 50 eps restore best，会显著拖慢且会“打断”学习轨迹。
        # 如果你确实想保留这种 hill-climb 行为，再把下面两行取消注释：
        # if (ep + 1) % 50 == 0:
        #     saver.restore(sess, best_path)

    # end training summary
    if num_episodes > 0:
        total_time = time.time() - train_start
        steps_per_sec = total_steps / max(total_time, 1e-6)
        print("total_steps = ", total_steps, "total_time_sec = ", total_time, "steps_per_sec = ", steps_per_sec)
        print("history best = ", history_best)

        saver.restore(sess, best_path)

    # -----------------------------
    # Evaluation (keep your logic)
    # -----------------------------
    def LOAD_MODEL(path_):
        fin = open(path_, "r")
        grid = int(fin.readline())
        aux = SM_env(max_hidden_block=20, attacker_fraction=0.4, follower_fraction=0.5, dev=0.0)
        policy = np.zeros((grid, aux._state_space_n), dtype=int)
        for i in range(grid):
            policy[i] = list(map(int, fin.readline().rstrip().split(' ')))
        fin.close()
        return policy

    optimal_policy_all = LOAD_MODEL("optimal_policy.txt")
    aux_env = SM_env(max_hidden_block=HIDDEN_BLOCK, attacker_fraction=ALPHA, follower_fraction=GAMMA)

    grid = 100
    rept = 1

    # OSM eval
    avg = 0.0
    for _ in range(rept):
        s = env.reset()
        for _ in range(10000):
            ss = aux_env._vector_to_index(s[0:3])
            a = optimal_policy_all[int(ALPHA * grid), ss]
            s, r, d, _ = env.step(s, int(a), move=True)
        avg += env.reward_fraction / rept
    print("OSM = ", avg)
    print("OSM = ", avg, file=file)

    # RL greedy eval (e=0)
    avg = 0.0
    for _ in range(rept):
        s_raw = env.reset()
        s_vec = np.asarray(s_raw, dtype=np.float32)
        for _ in range(10000):
            a = mainQN.act_epsilon_greedy(sess, s_vec, 0.0)
            s1_raw, r, d, _ = env.step(s_raw, int(a), move=True)
            s_raw = s1_raw
            s_vec = np.asarray(s_raw, dtype=np.float32)
        avg += env.reward_fraction / rept

    print("final simulation fraction = ", avg)
    print("final simulation fraction = ", avg, file=file)
