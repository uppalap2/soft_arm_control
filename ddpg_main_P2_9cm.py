
#from gym_torcs import TorcsEnv
import numpy as np
from numpy.linalg import norm
import random
import argparse
from math import sin, cos, pi, radians
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import scipy.io as sio
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
from matplotlib import pyplot as plt
import sys
import matlab.engine
from collections import deque

eng = matlab.engine.start_matlab() #Initialize MATLAB engine

OU = OU()  # Ornstein-Uhlenbeck Process


# Training data obtained from sparse grid

def plot_data(data):
    print('Plotting')
    plt.plot(data)
    plt.show()


def getPos(data_dict,L):
    Pb = round(np.random.uniform(11,40))
    Pr = round(np.random.uniform(-44,44))
    #theta = float(10 * round(float(np.random.uniform(-9, 9)) ))  # 120 deg sweep for turning
    point = data_dict[(float(Pb), float(Pr),L)]
    point = np.round(np.asarray(point), 2)
    point = np.reshape(point, [1, 3])
   # point.shape
    print("New training Point: {}, Pressure: {}".format(point[0], [Pb, Pr]))
    return point[0], Pb, Pr#,theta


def updateState(state, target, data_dict, a, Pb, Pr,L):#, th):
    pressure_flag = 0
    Pb_new = np.round( Pb + a[0], 1)#np.round(Pb + a[0], 1)
    Pr_new = np.round( Pr + a[1], 1)#np.round(Pr + a[1], 1)
   # th_new = np.round( th + a[2])#np.round(th + a[2])

    if (Pb_new) > 40:  # Bending
        Pb_new = 40
        pressure_flag = 1
    if (Pr_new) > 44:  # Rotating
        Pr_new = 44
        pressure_flag = 1
    if (Pb_new) < 11:
        Pb_new = 11
        pressure_flag = 1
    if (Pr_new) < -44:
        Pr_new = -44
        pressure_flag = 1
   # if (th_new) < -60:
       # th_new = -60
       # pressure_flag = 1
    #if (th_new) > 60:
        #th_new = 60
        #pressure_flag = 1

    new_point = eng.forward_kin(float(Pb_new), float(Pr_new),float(0),float(L))#, float(th_new),26e-2)
    new_point = np.round(np.asarray(new_point), 2)
    new_point = np.reshape(new_point, [1, 3])
    new_state = new_point[0] - target
    #print("new pressures:{}".format([Pb_new, Pr_new, th_new,pressure_flag]))
    #return new_state, Pb_new, Pr_new, th_new, pressure_flag
    return new_state, Pb_new, Pr_new, pressure_flag

def scaleActions(a_t):
    scaled_actions = np.zeros((1, 3))
    scaled_actions[0][0] = 1.0 * (-11 + 29 * (a_t[0][0] + 1) / 2.0)  #1.0 * (-9 + 18 * (a_t[0][0] + 1) / 2.0) 
    scaled_actions[0][1] = 1.0 * (-44 + 88 * (a_t[0][1] + 1) / 2.0)
    #scaled_actions[0][2] = 1.0 * (-60 + 120 * (a_t[0][2] + 1) / 2.0)
    #print("Scaled actions:{}".format(scaled_actions[0][0:3]))
    return scaled_actions


def calculateReward(target, state, state_new, PF, step):
    done = False
    reached = False
    reward = -1.0  
    err_prev = norm(state)
    err_curr = norm(state_new)
    reward += -2+(err_prev - err_curr)
    if PF:
        reward -= -2 + err_curr  # penalized by how far the crash occurs from target
        done = True

    if step > 100:
        reward -= -2 + err_curr
        done = True

    if err_curr < 1.0:
        done = True
        reached = True
        reward += 100
        print("TARGET ACQUIRED!")

    return reward, done, reached, err_curr


def playGame(train_indicator=0):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 20000
    BATCH_SIZE = 128
    GAMMA = 0.95
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.00002  # Learning rate for Actor
    LRC = 0.0001  # Learning rate for Critic

    action_dim = 2  # Bending/Rotating/Turning
    state_dim = 5  # of sensors input

    #    np.random.seed(1337)

    EXPLORE = 100000  # exploration step count
    episode_count = 100000
    rewards = 0
    done = False
    step = 0
    epsilon = 0.9 # Annealed to 0.1# was 0.9
    train_indicator = 1

    contents = sio.loadmat('P2_data_9cm.mat')
    data = contents['P2_data_9cm']
    data = tuple(map(tuple, data))
    data_dict = {}
    for i in data:
        data_dict[i[0:3]] = i[3:6]

    sess = tf.Session()
    from keras import backend as K #Keras with tf backend
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    done = False
    reached = False
    reached_count = 0
    rpe = []
    reached_deq = deque(maxlen=100)
    L = 9e-2;
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        step = 1

        start, Pb, Pr = getPos(data_dict,L)
        target, Pb_t, Pr_t = getPos(data_dict,L)
        state = start - target
        s_t = np.hstack((state[0], state[1], state[2], Pb, Pr))

        done = False
        total_reward = 0.0

        while not done:
            loss = 0
            if epsilon > 0.1:
                epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])
            #
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.50, 0.2)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.0, 0.50, 0.2)
            #noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.0, 0.50, 0.2)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            #a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            scaled_a = scaleActions(a_t)
           
            state_new, Pb_new, Pr_new, pressure_flag = updateState(s_t[0:3], target, data_dict, scaled_a[0],
                                                                           s_t[3], s_t[4], L)
            

            step += 1

            r_t, done, reached, err_curr = calculateReward(target, state, state_new, pressure_flag, step)
            s_t1 = np.hstack((state_new[0], state_new[1], state_new[2], Pb_new, Pr_new))

            if reached:
                reached_count += 1
            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[2] for e in batch])
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
            state = state_new
        reached_deq.append(reached)
        if np.mod(i, 3) == 0:
            if (train_indicator):
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        rpe.append(total_reward)
        sio.savemat('Rewards_per_episode', {'R': rpe})
        print(Pb, Pr)
        print(Pb_t, Pr_t)
        print(s_t1)
        print("Noise", np.round(noise_t, 2))
        print("Err", err_curr, "Reward ", total_reward, "Epsilon", epsilon)
        print("Total Step: " + str(step), "last 100 reached", np.sum(reached_deq))
        print("")


if __name__ == "__main__":
    playGame()
