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

class ReplayAgent(object):
    """This replays an optimal sequence of events"""
    ACTIONS = ['L', 'U', 'R', 'D']

    def __init__(self, action_space,playerno,memory=[]):
        self.action_space = action_space
        self.player_number = playerno
        self.MEMORY = memory
        self._actions = {value: i for i, value in enumerate(self.ACTIONS)}

    def act(self, step):
        if step >= 0 and step < len(self.MEMORY):
            optimal_action = self.MEMORY[step]
            return self._actions[optimal_action]
        else:
            raise Exception("out of sequence")

def one_debug():

    env = GoldMiner.GoldMinerEnv(players=1,totalgold=2,cooperative=False)
    episode_count = 1
    # with this seed instead it gets shot in the hostile field of the right
    env.seed(3)
    agent1 = ReplayAgent(env.action_space,1,['U','L','U','R','U','U','R','D','D'])
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        total_reward = 0
        steps = 0
        while steps < len(agent1.MEMORY):
            action = agent1.act(steps)
            ob, reward, done, info = env.step(action)
            total_reward += reward
            logger.warn('Game info %s : ' % info)
            if done:
                break
            steps+=1
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        logger.warn("Total reward in episode %d with %d steps is %.2f " % (i,steps,total_reward) )
    # Close the env and write monitor result info to disk
    env.close()
def one_optimal_player():
    env = GoldMiner.GoldMinerEnv(players=1,totalgold=2,cooperative=False)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '../multiagent/one-optimal'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    # with this seed the player is lucky so it gets 2.0 reward
    env.seed(1)
    agent1 = ReplayAgent(env.action_space,1,['U','L','U','R','D','D'])
    episode_count = 1
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        total_reward = 0
        steps = 0
        while steps < len(agent1.MEMORY):
            action = agent1.act(steps)
            ob, reward, done, info = env.step(action)
            total_reward += reward
            logger.warn('Game info %s : ' % info)
            if done:
                break
            steps+=1
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        logger.warn("Total reward in episode %d with %d steps is %.2f " % (i,steps,total_reward) )
    # Close the env and write monitor result info to disk
    env.close()

    # with this seed instead it gets shot in the hostile field of the right
    env.seed(3)
    agent1 = ReplayAgent(env.action_space,1,['U','L','U','R','D','D'])
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        total_reward = 0
        steps = 0
        while steps < len(agent1.MEMORY):
            action = agent1.act(steps)
            ob, reward, done, info = env.step(action)
            total_reward += reward
            logger.warn('Game info %s : ' % info)
            if done:
                break
            steps+=1
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        logger.warn("Total reward in episode %d with %d steps is %.2f " % (i,steps,total_reward) )
    # Close the env and write monitor result info to disk
    env.close()

    # with this seed instead it gets shot in the hostile field of the right
    env.seed(3)
    agent1 = ReplayAgent(env.action_space,1,['U','L','U','R','U','U','R','D','D'])
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        total_reward = 0
        steps = 0
        while steps < len(agent1.MEMORY):
            action = agent1.act(steps)
            ob, reward, done, info = env.step(action)
            total_reward += reward
            logger.warn('Game info %s : ' % info)
            if done:
                break
            steps+=1
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        logger.warn("Total reward in episode %d with %d steps is %.2f " % (i,steps,total_reward) )
    # Close the env and write monitor result info to disk
    env.close()

def one_qlearning_player():
    from gym.algo.qlearn import QLearn
    import numpy

    env = GoldMiner.GoldMinerEnv(players=1,totalgold=1,cooperative=False)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '../multiagent/one-qlearn'
    env = wrappers.Monitor(env, directory=outdir, force=True)

    number_of_features = env.observation_space.shape[0]
    last_time_steps = numpy.ndarray(0)

    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),alpha=0.5, gamma=0.90, epsilon=0.1)
    max_number_of_steps = 12
    for i_episode in range(3000):
        observation = env.reset()
        state = ''
        for t in range(max_number_of_steps):
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            nextState = ''

            # # If out of bounds
            # if (cart_position > 2.4 or cart_position < -2.4):
            #     reward = -200
            #     qlearn.learn(state, action, reward, nextState)
            #     print("Out of bounds, reseting")
            #     break

            if not(done):
                qlearn.learn(state, action, reward, nextState)
                state = nextState
            else:
                # Q-learn stuff
                reward = -200
                qlearn.learn(state, action, reward, nextState)
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break

    l = last_time_steps.tolist()
    l.sort()
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.monitor.close()

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
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    # Play first a one random player
    #one_random_player()
    #one_optimal_player()
    one_debug()
    #one_qlearning_player()
