import argparse
import sys

import gym
from gym import wrappers, logger
from gym.envs.multiplayer import GoldMiner

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space,playerno):
        self.action_space = action_space
        self.player_number = playerno

    def act(self, observation, reward, done):
        return self.action_space.sample()


def one_qlearning_player():
    from gym.algo.qlearn import QLearn
    import numpy

    env = GoldMiner.GoldMinerEnv(players=1,totalgold=1,cooperative=False,maxsteps=12)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '../multiagent/one-qlearn'
    env = wrappers.Monitor(env, directory=outdir, force=True)

    last_time_steps = numpy.ndarray(0)

    # The Q-learn algorithm
    qlearn = QLearn(observations=env.observation_space,actions=env.action_space,alpha=0.5, gamma=0.90, epsilon=0.1)
    space_size = qlearn.getObsSize()
    print("State space cardinality is %d " % space_size)

    max_number_of_steps = 12
    cumulated_reward = 0
    for i_episode in range(space_size+1):
        observation = env.reset()
        # Encode state to a compact string serialization
        state = qlearn.encodeState(observation)
        cumulated_reward = 0
        # decay epsilon if necessary
        qlearn.epsilon = qlearn.epsilon * 1.0
        print("Episode {:d} ".format(i_episode))
        for t in range(max_number_of_steps+1):
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            print("\tGame State Time {:d}: {:s}".format(t,str(info)))

            # Encode next state as well
            nextState = qlearn.encodeState(observation)

            qlearn.learn(state, action, reward, nextState)
            state = nextState
            cumulated_reward += reward

            if done:
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break

        print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))


def one_random_player():
    env = GoldMiner.GoldMinerEnv(players=1,totalgold=2,cooperative=False)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '../multiagent/one-random'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(5)
    agent1 = RandomAgent(env.action_space,1)
    episode_count = 1
    reward = 0
    done = False


    for i in range(episode_count):
        ob = env.reset()
        total_reward = 0
        steps = 0
        while steps < 24:
            action = agent1.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            total_reward += reward
            logger.warn('Game info %s : ' % info)
            if done:
                break
            steps+=1
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        logger.warn("Total reward in episode %d with %d steps is %d " % (i,steps,total_reward) )
    # Close the env and write monitor result info to disk
    env.close()


def two_random_compete_player():
    env = GoldMiner.GoldMinerEnv(players=2,totalgold=2,cooperative=False)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '../multiagent/two-random'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent1 = RandomAgent(env.action_space,1)
    agent2 = RandomAgent(env.action_space,2)

    episode_count = 1

    reward1 = 0
    done1 = False

    reward1 = 0
    done2 = False

    for i in range(episode_count):
        ob = env.reset()

        while True:

            if done1 == False:
                env.select_player(agent1.player_number)

                action = agent1.act(ob, reward1, done1)
                ob, reward1, done1, _ = env.step(action)

            if done2 == False:
                env.select_player(agent2.player_number)

                action = agent2.act(ob, reward2, done2)
                ob, reward2, done2, _ = env.step(action)

            if done1 and done2:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

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
