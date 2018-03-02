'''
Basic tabular Q-learning class
'''
import gym
import random
import pandas as pd
import numpy as np

class QLearn:
    def __init__(self, observations, actions, epsilon, alpha, gamma ,epsgreedy= True, tstart = 1000):
        '''
        Initialize the Q-learning variables and do some space calculations
        :param gym.spaces.Dict observations: gym space
        :param spaces.Discrete actions: gym space
        :param float epsilon: exploration constant
        :param float alpha: discount constant
        :param float gamma: discount factor
        :param bool epsgreedy: enable eps-greedy policy if True or Softmax is False
        '''
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.total_actions = actions.n  # actions available
        self.actions = range(actions.n) # action indexes
        self.epsgreedy = epsgreedy

        if self.epsgreedy == False:
            self.temperature = 1.0 * tstart
            self.tstep = 1.0

        if isinstance(observations,gym.spaces.Dict):
            self.obs_space = {}
            self.obs_space_size = 1
            self.obs_space_index = {}
            self.obs_space_rindex = {}
            for idx, item in enumerate(observations.spaces.items()):
                self.obs_space[item[0]] = item[1].n
                self.obs_space_size *= item[1].n
                self.obs_space_index[item[0]] = idx
                self.obs_space_rindex[idx] = item[0]
            self.obs_dims = len(observations.spaces.keys())
        else:
            raise Exception("Not implemented yet")

        self.glie = False

    def enableGLIE(self):
        '''
            Greedy in the Limit with Infinite Exploration
        :return:
        '''
        self.glie = True
        self.obs_space_visits = {}

    def setTau(self,tstart):
        '''
            Set the initial temperature for the Softmax exploration algorithm
        :param float tstart: the initial temperature parameter
        :return: void
        '''
        self.temperature = 1.0 * tstart

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
        :rtype: str
        '''
        ordered = {}
        for key,index in self.obs_space_index.items():
            ordered[index] = dict_state[key]

        return ":".join(str(v) for k, v in sorted(ordered.items()))

    def decodeState(self,state):
        '''
        Decode into the original observation from the simplified string
        :param str state: the string
        :return: the observation
        :rtype: dict
        '''
        values = state.split(':')
        obs = {}
        if len(values)==0:
            raise Exception('Wrong format')
        else:
            for index,value in enumerate(values):
                obs[self.obs_space_rindex[index]] = value

        return obs

    def exportToPandas(self):
        rows = []
        for state_action,value in self.q.items():
            row = self.decodeState(state_action[0])
            row['action'] = state_action[1]
            row['value'] = value
            rows.append(row)

        return pd.DataFrame(rows)

    def loadQ(self,dict):
        self.q = dict

    def getQ(self, state, action):
        '''
        Return the value associated with the state-action pair
        :param state: an encoded gym state
        :param action: an integer from discrete gym state
        :return: the value
        :rtype: float
        '''
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

    def weighted_choice(self,weights):
        # cun sum in ascending order
        totals = np.cumsum(weights)
        # this should be 1.0
        norm = totals[-1]
        # draw a random number between 0.0 and 1.0
        draw = np.random.rand() * norm
        return np.searchsorted(totals, draw)

    def chooseAction(self, state, return_q=False):
        '''
        Choose the best action
        :param state: an integer from the gym discrete space
        :param return_q: if True returns also the value, if False only the action
        :return: the action and or the value of the state
        '''
        q = [self.getQ(state, a) for a in self.actions]

        if self.epsgreedy:
            maxQ = max(q)

            if self.glie:
                # reduce probability if GLIE is enabled
                if state in self.obs_space_visits:
                    maxp = self.epsilon / self.obs_space_visits[state]
                else:
                    maxp = self.epsilon
            else:
                maxp = self.epsilon

            if random.random() < maxp:
                minQ = min(q);
                # magnitude as absolute difference between min and max Q
                mag = max(abs(minQ), abs(maxQ))
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

            #update state visits if necessary
            if self.glie:
                if state in self.obs_space_visits:
                    self.obs_space_visits[state] += 1
                else:
                    self.obs_space_visits[state] = 1

            if return_q: # if they want it, give it!
                return action, q
            return action
        else:
            prob_t = np.zeros((len(self.actions),), dtype=float)

            if self.temperature <= 0:
                self.temperature += self.tstep

            for actionid,qvalue in enumerate(q):
                prob_t[actionid] = np.exp(qvalue/self.temperature)

            prob_t = np.divide(prob_t,sum(prob_t))

            action = self.weighted_choice(prob_t)

            self.temperature -= self.tstep

            if return_q: # if they want it, give it!
                return action, prob_t
            else:
                return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)