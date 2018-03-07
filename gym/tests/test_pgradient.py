from gym.algo.pgradient import PolicyGradientAgent
from gym import wrappers, logger
from gym import spaces
from collections import Counter
import tensorflow as tf

class TestQLearner(object):

    def test_tf(self):
        with tf.Graph().as_default(), tf.Session() as sess:
            assert True == True