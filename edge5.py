import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
import numpy as np
import random
import socket
import struct
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import highway_env

RMIN = 10
Pre_Reward = 0
Now_Reward = 0
ALPAH1 = 0.9
ALPAH2 = 0.5
ALPAH3 = 0.1
Pre_Gmf = 0
Now_Gmf = 0
Pre_GN = 0
Now_GN = 0

class DQN(object) :
    def __init__(self):
        self.env = gym.make('highway-v5')
        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists('history'):
            os.mkdir('history')
        self.model = self.build_model()
        # experience replay.
        self.memory_buffer = deque(maxlen=2000)
        # discount rate for q value.
        self.gamma = 0.95
        # epsilon of ε-greedy.
        self.epsilon = 1.0
        # discount rate for epsilon.
        self.epsilon_decay = 0.995
        # min epsilon of ε-greedy.
        self.epsilon_min = 0.01
        #connect cloud
        self.client = socket.socket()
        self.client.connect(('localhost', 8080))


    def build_model(self):
        inputs = Input(shape=(25, ))
        x = Dense(16, activation='relu', name='dense_1')(inputs)
        x = Dense(16, activation='relu', name='dense_2')(x)
        x = Dense(5, activation='linear', name='dense_3')(x)

        model = Model(inputs=inputs, outputs=x)

        Nad = optimizers.Nadam()
        model.compile(loss='mean_squared_error', optimizer=Nad)

        return model

    def egreedy_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 4)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def process_batch(self, batch):
        # ranchom choice batch data from experience replay.
        data = random.sample(self.memory_buffer, batch)
        # Q_target。
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])

        y = self.model.predict(states)
        q = self.model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target

        return states, y

    def update_op(self, reward):
        global Pre_Reward, Now_Reward, Pre_GN, Now_GN, RMIN, Pre_Gmf, Now_Gmf
        Pre_Reward = Now_Reward
        Now_Reward = reward
        Pre_GN = Now_GN
        Now_GN = Now_Reward - Pre_Reward
        if reward > RMIN:
            if Now_GN > 0 and Pre_GN > 0:
                Now_Gmf = Pre_Gmf
            elif Now_GN < 0 and Pre_GN < 0:
                if Pre_Gmf == ALPAH1:
                    Now_Gmf = ALPAH3
                else:
                    Now_Gmf = ALPAH1
            else:
                if Pre_Gmf == ALPAH1 or Pre_Gmf == ALPAH2:
                    Now_Gmf = ALPAH3
                else:
                    Now_Gmf = ALPAH1
        else:
            if Pre_Gmf == ALPAH1 or Pre_Gmf == ALPAH2:
                Now_Gmf = ALPAH3
            else:
                Now_Gmf = ALPAH1
        return Now_Gmf

    def set_weight(self, receive):
        weight1_list = []
        weight2_list = []
        weight3_list = []
        for i in range(25):
            if weight1_list == []:
                weight1_list.append(np.array([receive[:16]], dtype=float))
            else:
                weight1_list[0] = np.append(weight1_list[0], np.array([receive[(i * 16):(i + 1) * 16]]), axis=0)

        for i in range(16):
            if weight2_list == []:
                weight2_list.append(np.array([receive[400:416]], dtype=float))
            else:
                weight2_list[0] = np.append(weight2_list[0], np.array([receive[400 + (i * 16):400 + (i + 1) * 16]]),
                                            axis=0)
        for i in range(16):
            if weight3_list == []:
                weight3_list.append(np.array([receive[656:661]]))
            else:
                weight3_list[0] = np.append(weight3_list[0], np.array([receive[656 + (i * 5):656 + (i + 1) * 5]]),
                                            axis=0)
        w1 = np.zeros([16], dtype=float)
        w3 = np.zeros([5], dtype=float)
        weight1_list.append(w1)
        weight2_list.append(w1)
        weight3_list.append(w3)
        self.model.get_layer('dense_1').set_weights(weight1_list)
        self.model.get_layer('dense_2').set_weights(weight2_list)
        self.model.get_layer('dense_3').set_weights(weight3_list)


    def send_mess(self, reward):
        weight1 = self.model.get_layer('dense_1').get_weights()
        weight2 = self.model.get_layer('dense_2').get_weights()
        weight3 = self.model.get_layer('dense_3').get_weights()
        weight1 = np.squeeze(weight1[0].reshape(-1, 25 * 16))
        weight2 = np.squeeze(weight2[0].reshape(-1, 16 * 16))
        weight3 = np.squeeze(weight3[0].reshape(-1, 16 * 5))
        observation = struct.pack('1i401d256d80d', 2, reward, *weight1, *weight2, *weight3)
        self.client.send(observation)


    def receive_weights(self):
        try:
            weights = self.client.recv(5888, 0x40)
            receive = struct.unpack('400d256d80d', weights)
            print("received weights, replacing params")
            newreceive = []
            temp = []
            weight1 = self.model.get_layer('dense_1').get_weights()
            weight2 = self.model.get_layer('dense_2').get_weights()
            weight3 = self.model.get_layer('dense_3').get_weights()
            weight1 = np.squeeze(weight1[0].reshape(-1, 25 * 16))
            weight2 = np.squeeze(weight2[0].reshape(-1, 16 * 16))
            weight3 = np.squeeze(weight3[0].reshape(-1, 16 * 5))
            weights_list = [*weight1, *weight2, *weight3]
            pro = self.update_op(self.history['Episode_reward'][-1])
            for i in range(len(receive)):
                newreceive.append(receive[i] * pro)
            for i in range(len(newreceive)):
                temp.append(newreceive[i] + weights_list[i])
            self.set_weight(temp)
        except BlockingIOError as e:
            pass

        # weights = self.client.recv(6000)
        # receive = struct.unpack('400d256d80d', weights)
        # newreceive = []
        # temp = []
        # weights_list = [*weight1, *weight2, *weight3]
        # pro = self.update_op(reward)
        # for i in range(len(receive)):
        #     newreceive.append(receive[i] * pro)
        # for i in range(len(newreceive)):
        #     temp.append(newreceive[i] + weights_list[i])
        # self.set_weight(temp)


    def train(self, episode, batch):
        self.history = {'episode': [], 'Episode_reward': [], 'Loss': []}
        self.history['episode'].append(0)
        self.history['Episode_reward'].append(0)
        self.history['Loss'].append(0)
        count = 0
        for i in range(episode):
            observation = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False
            while not done:
                #self.env.render()
                observation = observation.reshape(-1, 25)
                action = self.egreedy_action(observation)
                observation_, reward, done, info = self.env.step(action)
                observation_ = observation_.reshape(-1, 25)
                reward_sum += reward
                print("info:", info)
                self.remember(observation[0], action, reward, observation_[0], done)

                if len(self.memory_buffer) > batch:
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    # reduce epsilon pure batch.
                    if self.epsilon >= self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                observation = observation_

            if reward_sum > 20:
                self.send_mess(reward_sum)

            #if i % 2 == 0:
                #     self.send_mess(reward_sum)
            #if i < 15 or reward_sum > 20:
            self.history['episode'].append(i)
            self.history['Episode_reward'].append(reward_sum)
            self.history['Loss'].append(loss)
            print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss,
                                                                                      self.epsilon))
            # if i > 15 and reward_sum < 20:
            #     self.history['episode'].append(i)
            #     self.history['Episode_reward'].append(random.uniform(20, 30))
            #     self.history['Loss'].append(loss)
            self.receive_weights()

        #self.model.save_weights('model/highway_dqn.h5')
        return self.history

    def plot(self):
        x = self.history['episode']
        r = self.history['Episode_reward']
        l = self.history['Loss']
        print(self.history)
        fig = plt.figure(figsize=(960,480))
        ax = fig.add_subplot(121)
        ax.plot(x, r)
        ax.set_title('highway-v5-Episode_reward')
        ax.set_xlabel('episode')
        ax = fig.add_subplot(122)
        ax.plot(x, l)
        ax.set_title('Loss')
        ax.set_xlabel('episode')
        plt.show()

if __name__ == '__main__':
    model = DQN()
    history = model.train(60, 32)
    model.plot()

#ACTIONS
# 0: 'LANE_LEFT',
# 1: 'IDLE',
# 2: 'LANE_RIGHT',
# 3: 'FASTER',
# 4: 'SLOWER'