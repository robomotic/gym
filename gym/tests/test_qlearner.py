from gym.algo.qlearn import QLearn
from gym import wrappers, logger
from gym import spaces

class TestQLearner(object):

    def test_eps_greedy(self):
        ACTIONS = ['L', 'U', 'R', 'D']
        POSITIONS = ['Home', 'Free', 'HostileL', 'HostileR', 'Gold']

        action_space = spaces.Discrete(len(ACTIONS))
        position_space = spaces.Dict({"Position":spaces.Discrete(len(POSITIONS))})

        qlearn = QLearn(observations=position_space,actions=action_space,alpha=1.0, gamma=0.0, epsilon=0.0,epsgreedy=True)
        qlearn.seed(0)
        best_action_state = 1
        for posidx,position in enumerate(POSITIONS):
            for actid,action in enumerate(ACTIONS):
                if actid == best_action_state:
                    qlearn.q[(posidx, actid)] = 2.0
                else:
                    qlearn.q[(posidx, actid)] = 1.0

        for repeat in range(1,100):
            action,q = qlearn.chooseAction(state=0,return_q=True)
            assert action == best_action_state
            assert min(q) == 1.0
            assert max(q) == 2.0

        qlearn = QLearn(observations=position_space,actions=action_space,alpha=1.0, gamma=0.0, epsilon=1.0,epsgreedy=True)
        qlearn.seed(0)
        best_action_state = 1
        for posidx,position in enumerate(POSITIONS):
            for actid,action in enumerate(ACTIONS):
                if actid == best_action_state:
                    qlearn.q[(posidx, actid)] = 2.0
                else:
                    qlearn.q[(posidx, actid)] = 1.0
        actions = []
        for repeat in range(1,100):
            action,q = qlearn.chooseAction(state=0,return_q=True)
            actions.append(action)

        assert len(set(actions)) == len(ACTIONS)

    def test_softmax(self):
        ACTIONS = ['L', 'U', 'R', 'D']
        POSITIONS = ['Home', 'Free', 'HostileL', 'HostileR', 'Gold']

        action_space = spaces.Discrete(len(ACTIONS))
        position_space = spaces.Dict({"Position":spaces.Discrete(len(POSITIONS))})

        qlearn = QLearn(observations=position_space,actions=action_space,alpha=1.0, gamma=0.0, epsilon=0.0,epsgreedy=False)
        qlearn.seed(0)
        best_action_state = 1
        for posidx,position in enumerate(POSITIONS):
            for actid,action in enumerate(ACTIONS):
                if actid == best_action_state:
                    qlearn.q[(posidx, actid)] = 1.5
                else:
                    qlearn.q[(posidx, actid)] = 1.0
        qlearn.setTau(1000)
        actions = []
        for repeat in range(1,1000):
            action,prob_t = qlearn.chooseAction(state=0,return_q=True)
            actions.append(action)
        print("Total actions explored %d" % len(set(actions)))
        assert len(set(actions)) == 4

