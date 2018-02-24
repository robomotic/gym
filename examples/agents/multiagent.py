import argparse
import sys

import gym
from gym import wrappers, logger
from gym.envs.multiplayer import GoldMiner

import pprint

def optimize_one_qlearning_player():
    pass

def one_qlearning_player():
    from gym.algo.qlearn import QLearn
    import numpy as np
    from collections import Counter

    max_steps = 18

    env = GoldMiner.GoldMinerEnv(players=1,totalgold=1,maxsteps=max_steps)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '../multiagent/one-qlearn'
    env = wrappers.Monitor(env, directory=outdir, force=True)

    last_time_steps = np.ndarray(0)

    # The Q-learn algorithm: alpha is the learning rate, gamma is the discount reward factor and epsilon is the exploration constant
    qlearn = QLearn(observations=env.observation_space,actions=env.action_space,alpha=0.25, gamma=0.90, epsilon=0.1)

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
    parser.add_argument('env_id', nargs='?', default='GoldMiner-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    # Play first a one random player
    one_qlearning_player()
