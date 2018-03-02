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
from PIL import Image
import json
import pickle

def test_graphic():
    from gym.envs.classic_control import rendering
    screen_width = 400
    screen_height = 800

    viewer = rendering.Viewer(screen_width, screen_height)
    viewer.set_bounds(-2.0, 2.0, -4.0, 4.0)

    gold_xy =(0.0,3.0)
    home_xy = (0.0,-3.0)
    free_xy = (0.0,-1.0)
    hostile_lxy = (-1.0,1.0)
    hostile_rxy = (1.0,1.0)

    gold_transform = rendering.Transform(rotation=0, translation=gold_xy)
    circ = viewer.draw_circle(0.4)
    circ.set_color(1.0, 1.0, 0)
    circ.add_attr(gold_transform)

    line_gold_hostiler = viewer.draw_line(start=gold_xy,end=hostile_rxy)
    line_gold_hostiler.set_color(1.0,0.0,0.0)
    line_gold_hostiler.add_attr(rendering.LineWidth(10))

    line_gold_hostilel = viewer.draw_line(start=gold_xy,end=hostile_lxy)
    line_gold_hostilel.set_color(1.0,0.0,0.0)
    line_gold_hostilel.add_attr(rendering.LineWidth(10))

    home_transform = rendering.Transform(rotation=0, translation=home_xy)
    circ = viewer.draw_circle(0.4)
    circ.set_color(.0, .0, 1.0)
    circ.add_attr(home_transform)

    free_transform = rendering.Transform(rotation=0, translation=free_xy)
    circ = viewer.draw_circle(0.4)
    circ.set_color(.0, 1.0, 0.0)
    circ.add_attr(free_transform)

    line_home_free = viewer.draw_line(start=home_xy,end=free_xy)
    line_home_free.set_color(1.0,0.0,0.0)
    line_home_free.add_attr(rendering.LineWidth(10))

    hostile_ltransform = rendering.Transform(rotation=0, translation=hostile_lxy)
    circ = viewer.draw_circle(0.4)
    circ.set_color(1.0, .0, .0)
    circ.add_attr(hostile_ltransform)

    line_free_hostilel = viewer.draw_line(start=free_xy,end=hostile_lxy)
    line_free_hostilel.set_color(1.0,0.0,0.0)
    line_free_hostilel.add_attr(rendering.LineWidth(10))

    hostile_rtransform = rendering.Transform(rotation=0, translation=hostile_rxy)
    circ = viewer.draw_circle(0.4)
    circ.set_color(1.0, .0, .0)
    circ.add_attr(hostile_rtransform)

    line_free_hostiler = viewer.draw_line(start=free_xy,end=hostile_rxy)
    line_free_hostiler.set_color(1.0,0.0,0.0)
    line_free_hostiler.add_attr(rendering.LineWidth(10))

    # draw player 1 for example

    player_circle = rendering.Transform(rotation=0, translation=(0.15,3.0))
    circ = viewer.draw_circle(0.2)
    circ.set_color(1.0, 1.0, 1.0)
    circ.add_attr(player_circle)

    viewer.draw_label(text='P1',position=home_xy)
    viewer.draw_label(text='P2', position=hostile_lxy)
    viewer.draw_label(text='Gold = 3', position=(-1.8,3.8))
    gamepic = viewer.render(True)

    img = Image.fromarray(gamepic, 'RGB')
    img.save('goldminergame.png')
    img.show()

def evaluate_qlearning_epsgreedy(args):
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


def evaluate_qlearning_softmax(args):
    index, params = args
    print('Thread {0} Evaluating params: {1}'.format(os.getpid(),params))

    max_steps = 18

    env = GoldMiner.GoldMinerEnv(players=1,totalgold=1,maxsteps=max_steps)

    # The Q-learn algorithm: alpha is the learning rate, gamma is the discount reward factor and epsilon is the exploration constant
    qlearn = QLearn(observations=env.observation_space,actions=env.action_space,alpha=params['alpha'], gamma=params['gamma'],epsilon=params['epsilon'], epsgreedy = False)
    qlearn.setTau(params['taustart'])
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

def optimize_one_qlearning_player(epsgreedy=True):

    folder_data = os.path.dirname(os.path.realpath(__file__))
    folder_data = os.path.join(folder_data,'parameters')

    if not os.path.exists(folder_data):
        os.makedirs(folder_data)

    if epsgreedy:
        grid_params = {
            'alpha': np.linspace(0.1, 0.5, num=10),
            'gamma': np.linspace(0.0, 1.0, num=10),
            'epsilon': np.linspace(0.1, 0.6, num=10)
        }
    else:
        grid_params = {
            'alpha': np.linspace(0.1, 0.5, num=10),
            'gamma': np.linspace(0.0, 1.0, num=10),
            'epsilon': np.array([0.0]),
            'taustart': np.linspace(10, 24, num=14)
        }
    grid = list(ParameterGrid(grid_params))

    start_time = time.time()
    cpu_count = multiprocessing.cpu_count()
    print('About to evaluate {0:d} parameter sets on {1:d} CPU'.format(len(grid),cpu_count))
    pool = multiprocessing.Pool(processes=cpu_count)

    if epsgreedy:
        final_scores = pool.map(evaluate_qlearning_epsgreedy, list(enumerate(grid)))
    else:
        final_scores = pool.map(evaluate_qlearning_softmax, list(enumerate(grid)))

    print('Worse parameter set was {} with score of {}'.format(grid[np.argmin(final_scores)], np.min(final_scores)))
    print('Best parameter set was {} with score of {}'.format(grid[np.argmax(final_scores)], np.max(final_scores)))

    print('Execution time: {} sec'.format(time.time() - start_time))

    with open(os.path.join(folder_data,'epsgreedy.json' if epsgreedy else 'softmax.json'), 'w') as fp:
        summary = {'best':grid[np.argmax(final_scores)],'score':np.max(final_scores)}
        json.dump(summary, fp)


def one_qlearning_player(epsgreedy=True,max_episodes = 100000,render=True):
    '''
    Load the best parameters and run the agent
    :return: None
    '''
    folder = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(folder, 'parameters')
    max_steps = 30
    env = GoldMiner.GoldMinerEnv(players=1, totalgold=1, maxsteps=max_steps)
    if epsgreedy:
        fileparam = os.path.join(folder,'epsgreedy.json')

        with open(fileparam) as data_file:
            best = json.load(data_file)['best']

        # The Q-learn algorithm with the epsgreedy exploration
        qlearn = QLearn(observations=env.observation_space, actions=env.action_space, alpha=best['alpha'], gamma=best['gamma'],
                        epsilon=best['epsilon'])
    else:
        fileparam = os.path.join(folder,'softmax.json')

        with open(fileparam) as data_file:
            best = json.load(data_file)['best']

        # The Q-learn algorithm with softmax exploration
        qlearn = QLearn(observations=env.observation_space, actions=env.action_space, alpha=best['alpha'], gamma=best['gamma'],
                        epsilon=best['epsilon'])

    if render == True:
        with open(os.path.join(folder,'qtable-%s.dat' % 'epsgreedy' if epsgreedy else 'softmax'), 'rb') as handle:
            qtable = pickle.load(handle)
            qlearn.loadQ(qtable)

    outdir = '../multiagent/one-qlearn'
    monitor = wrappers.Monitor(env, directory=outdir, force=True)

    last_time_steps = np.ndarray(0)

    if render:
        # with this parameters the agent takes 2.0 at home
        env.seed(0)
        qlearn.seed(0)
        # with these parameters the agent get shot and takes 2.0
        env.seed(3)
        qlearn.seed(3)

    space_size = qlearn.getObsSize()

    print("State space cardinality is %d " % space_size)

    print_state = False
    debug_episode = 92

    episode_reward = []
    max_episodes = min(max_episodes,space_size*2)

    for i_episode in range(max_episodes):
        observation = monitor.reset()
        # Encode state to a compact string serialization
        state = qlearn.encodeState(observation)
        cumulated_reward = 0
        # decay epsilon if necessary
        qlearn.epsilon = qlearn.epsilon * 1.0

        if print_state or debug_episode == i_episode:
            print("Episode {:d} ".format(i_episode))

        for t in range(max_steps+1):
            if render:
                monitor.render(mode='rgb_array')
                fps = env.metadata.get('video.frames_per_second')
                time.sleep(1.0 / fps)
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = monitor.step(action)

            # Encode next state as well
            nextState = qlearn.encodeState(observation)

            if render == False:
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

    if render == False:
        with open(os.path.join(folder,'qtable-%s.dat' % 'epsgreedy' if epsgreedy else 'softmax'), 'wb') as handle:
            pickle.dump(qlearn.q, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # display simple histogram count
    histogram = Counter(episode_reward)

    for bin in histogram:
        print("Total episodes = {:d} with reward = {:f}".format(histogram[bin],bin))

    if render == False:
        # Export the Q table as panda frame
        qdf = qlearn.exportToPandas()
        # Replace state and actions
        qdf['action']=qdf['action'].map(lambda idx:env.ACTIONS[idx])
        qdf['Player1Position'] = qdf['Player1Position'].map(lambda idx: env.POSITIONS[int(idx)])

        qdf = qdf.loc[qdf['value'] > 0.0]

        # Save as HTML table
        #dphtml = qdf.style
        dphtml = qdf.to_html()

        with open(os.path.join(folder,'qtable-%s.html' % 'epsgreedy' if epsgreedy else 'softmax'), 'w') as f:
            f.write(dphtml)
            f.close()

        pivot1 = qdf.pivot_table(qdf, index=["Player1Position", "action","Player1HasGold","GoldHome"])
        pivot1.to_html(os.path.join(folder,'qpivot-%s.html' % 'epsgreedy' if epsgreedy else 'softmax'))

    # Close the env and write monitor result info to disk
    env.close()
    monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--best', dest='best', action='store_true',default = False)
    parser.add_argument('--epsgreedy', dest='epson', action='store_true', default=False)
    parser.add_argument('--grafix', dest='ui', action='store_true', default = False)
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    if args.best:
        # Play the policy
        if args.ui:
            one_qlearning_player(epsgreedy=args.epson,max_episodes=1,render = args.ui)
        else:
            one_qlearning_player(epsgreedy=args.epson,render = False)
    elif args.optimize:
        # Parameter search for optimal Q-learning player
        optimize_one_qlearning_player(args.epson)
    elif args.ui:
        test_graphic()
    else:
        parser.print_help()
        sys.exit(1)
