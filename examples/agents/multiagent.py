import argparse
import sys

import gym
from gym import wrappers, logger
from gym.envs.multiplayer import GoldMiner
import numpy as np
import pprint
from sklearn.model_selection import ParameterGrid
import time
import multiprocessing
from gym.algo.qlearn import QLearn
from collections import Counter,deque
import os

def evaluate_qlearning(args):
    index, params = args
    print('Thread {0} Evaluating params: {1}'.format(os.getpid(),params))

    max_steps = 18

    env = GoldMiner.GoldMinerEnv(players=1,totalgold=1,maxsteps=max_steps)

    # The Q-learn algorithm: alpha is the learning rate, gamma is the discount reward factor and epsilon is the exploration constant
    qlearn = QLearn(observations=env.observation_space,actions=env.action_space,alpha=params['alpha'], gamma=params['gamma'], epsilon=params['epsilon'])

    space_size = qlearn.getObsSize()

    episode_reward = []

    for i_episode in range(space_size*2):
        observation = env.reset()
        # Encode state to a compact string serialization
        state = qlearn.encodeState(observation)
        cumulated_reward = 0
        # decay epsilon if necessary
        qlearn.epsilon = qlearn.epsilon * 1.0

        for t in range(max_steps+1):

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Encode next state as well
            nextState = qlearn.encodeState(observation)

            qlearn.learn(state, action, reward, nextState)
            state = nextState
            cumulated_reward += reward

            if done:
                break

        episode_reward.append(cumulated_reward)

    # Close the env and write monitor result info to disk
    env.close()
    # With the mean agents that had shorter runs should get a higher score
    score = np.mean(episode_reward)

    return score

def optimize_one_qlearning_player():

    grid_params = {
        'alpha': np.linspace(0.1, 0.5, num=10),
        'gamma': np.linspace(0.0, 1.0, num=10),
        'epsilon': np.linspace(0.1, 0.6, num=10)
    }

    grid = list(ParameterGrid(grid_params))

    start_time = time.time()
    cpu_count = multiprocessing.cpu_count()
    print('About to evaluate {0:d} parameter sets on {1:d} CPU'.format(len(grid),cpu_count))
    pool = multiprocessing.Pool(processes=cpu_count)
    final_scores = pool.map(evaluate_qlearning, list(enumerate(grid)))

    print('Worse parameter set was {} with score of {}'.format(grid[np.argmin(final_scores)], np.min(final_scores)))
    print('Best parameter set was {} with score of {}'.format(grid[np.argmax(final_scores)], np.max(final_scores)))
    print('Execution time: {} sec'.format(time.time() - start_time))

def one_qlearning_player():
    '''
    It uses the parameters found by running optimize_one_qlearning_player
    Worse parameter set was {'alpha': 0.18888888888888888, 'epsilon': 0.4888888888888888, 'gamma': 0.0} with score of 0.890625
    Best parameter set was {'alpha': 0.2777777777777778, 'epsilon': 0.2111111111111111, 'gamma': 1.0} with score of 2.040625
    Execution time: 42.154309034347534 sec
    :return: None
    '''

    max_steps = 18

    env = GoldMiner.GoldMinerEnv(players=1,totalgold=1,maxsteps=max_steps)

    outdir = '../multiagent/one-qlearn'
    env = wrappers.Monitor(env, directory=outdir, force=True)

    last_time_steps = np.ndarray(0)

    # The Q-learn algorithm: alpha is the learning rate, gamma is the discount reward factor and epsilon is the exploration constant
    qlearn = QLearn(observations=env.observation_space,actions=env.action_space,alpha=0.27, gamma=1.0, epsilon=0.21)

    # make deterministic both environment and agent
    env.seed(0)
    qlearn.seed(0)

    space_size = qlearn.getObsSize()

    print("State space cardinality is %d " % space_size)

    print_state = False
    debug_episode = 92

    episode_reward = []

    for i_episode in range(space_size*2):
        observation = env.reset()
        # Encode state to a compact string serialization
        state = qlearn.encodeState(observation)
        cumulated_reward = 0
        # decay epsilon if necessary
        qlearn.epsilon = qlearn.epsilon * 1.0

        if print_state or debug_episode == i_episode:
            print("Episode {:d} ".format(i_episode))

        for t in range(max_steps+1):
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Encode next state as well
            nextState = qlearn.encodeState(observation)

            qlearn.learn(state, action, reward, nextState)
            state = nextState
            cumulated_reward += reward

            if print_state or debug_episode == i_episode:
                info['Time'] = t
                print("\tGame State:\n\t{:s}".format(str(info)))

            if done:
                last_time_steps = np.append(last_time_steps, [int(t + 1)])
                break

        episode_reward.append(cumulated_reward)

        print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))

    # display simple histogram count
    histogram = Counter(episode_reward)

    for bin in histogram:
        print("Total episodes = {:d} with reward = {:f}".format(histogram[bin],bin))

    # Close the env and write monitor result info to disk
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--best', dest='best', action='store_true',default = True)
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    if args.best:
        # Learn with a tabular Q-learning player
        one_qlearning_player()
    elif args.optimize:
        # Parameter search for optimal Q-learning player
        optimize_one_qlearning_player()
    else:
        args.print_help()
        sys.exit(1)
