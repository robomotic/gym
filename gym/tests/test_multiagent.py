from gym.envs.multiplayer import GoldMiner
from gym import wrappers, logger


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

class TestMultiPlayer(object):

    def test_optimal_player(self):
        logger.set_level(logger.WARN)

        env = GoldMiner.GoldMinerEnv(players=1, totalgold=2, cooperative=False)

        # You provide the directory to write to (can be an existing
        # directory, including one with existing data -- all monitor files
        # will be namespaced). You can also dump to a tempdir if you'd
        # like: tempfile.mkdtemp().
        outdir = '../multiagent/one-optimal'
        env = wrappers.Monitor(env, directory=outdir, force=True)
        # with this seed the player is lucky so it gets 2.0 reward
        env.seed(1)
        agent1 = ReplayAgent(env.action_space, 1, ['U', 'L', 'U', 'R', 'D', 'D'])
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
                steps += 1
                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.
            logger.warn("Total reward in episode %d with %d steps is %.2f " % (i, steps, total_reward))
        # Close the env and write monitor result info to disk
        env.close()
        assert total_reward == 2.0
        assert steps == 6

    def test_lost_gold(self):
        logger.set_level(logger.WARN)
        env = GoldMiner.GoldMinerEnv(players=1, totalgold=2, cooperative=False)

        # with this seed instead it gets shot in the hostile field of the right
        env.seed(3)
        # but it doesn't recover it
        agent1 = ReplayAgent(env.action_space, 1, ['U', 'L', 'U', 'R', 'D', 'D'])
        reward = 0
        done = False
        episode_count = 1
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
                steps += 1
                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.
            logger.warn("Total reward in episode %d with %d steps is %.2f " % (i, steps, total_reward))
        # Close the env and write monitor result info to disk
        env.close()

        assert total_reward == 0.5
        assert steps == 6

    def test_gold_retake(self):
        logger.set_level(logger.WARN)
        env = GoldMiner.GoldMinerEnv(players=1, totalgold=2, cooperative=False)
        episode_count = 1
        # with this seed instead it gets shot in the hostile field of the right
        env.seed(3)
        # but then it takes it back
        agent1 = ReplayAgent(env.action_space, 1, ['U', 'L', 'U', 'R', 'U', 'U', 'R', 'D', 'D'])
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
                steps += 1
                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.
            logger.warn("Total reward in episode %d with %d steps is %.2f " % (i, steps, total_reward))
        # Close the env and write monitor result info to disk
        env.close()

        assert total_reward == 2.0
        assert steps == 9