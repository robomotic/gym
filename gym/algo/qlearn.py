'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy
import random
from collections import OrderedDict

class QLearn:
    def __init__(self, observations, actions, epsilon, alpha, gamma):
        '''
        Initialize the Q-learning variables and do some space calculations
        :param gym.spaces.Dict observations: gym space
        :param spaces.Discrete actions: gym space
        :param float epsilon: exploration constant
        :param float alpha: discount constant
        :param float gamma: discount factor
        '''
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.total_actions = actions.n  # actions available
        self.actions = range(actions.n) # action indexes

        if isinstance(observations,gym.spaces.Dict):
            self.obs_space = {}
            self.obs_space_size = 1
            self.obs_space_index = {}
            for idx, item in enumerate(observations.spaces.items()):
                self.obs_space[item[0]] = item[1].n
                self.obs_space_size *= item[1].n
                self.obs_space_index[item[0]] = idx

            self.obs_dims = len(observations.spaces.keys())
        else:
            raise Exception("Not implemented yet")

    def seed(self,seed):
        random.seed(seed)

    def getObsSize(self):
        '''
        Gets the total dimension of the observation space calculated at init
        :return: the total number of states of the observations
        :rtype: int
        '''
        return self.obs_space_size

    def encodeState(self,dict_state):
        '''
        Encode the observation which is a dictionary in a simplified string
        :param Dict dict_state: the observation dictionary
        :return: a string representation of the dictionary
        '''
        ordered = {}
        for key,index in self.obs_space_index.items():
            ordered[index] = dict_state[key]

        return ":".join(str(v) for k, v in sorted(ordered.items()))

    def decodeState(self,state):
        pass

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)