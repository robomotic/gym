import sys
from six import StringIO, b

from gym import utils
from gym import Env, spaces
from gym.utils import seeding
import os

__copyright__ = "Copyright 2018, Paolo Di Prodi"
__credits__ = ["Me"]
__license__ = "BSD 3-Clause"
__author__ = "Paolo Di Prodi <paolo@robomotic.com>"

class GoldMinerEnv(Env):
    """

    """

    metadata = {'render.modes': ['human', 'ansi','rgb_array'],
                'video.frames_per_second' : 1}

    ACTIONS = ['L', 'U', 'R', 'D']
    POSITIONS = ['Home', 'Free', 'HostileL', 'HostileR', 'Gold']

    def __init__(self, players=1,totalgold= 1,maxsteps=100):

        # count the steps so far
        self.steps = 0
        # terminate when this count is reached
        self.maxsteps = maxsteps

        # set the path directory to load the state representation
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # set the possible actions
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        with open(os.path.join(dir_path,'board.txt'), 'r') as board_file:
            # read the ASCII art
            self._ascii_grid = board_file.read()

        # the total number of players
        self.players = players

        self._positions = {value:i for i,value in enumerate(self.POSITIONS)}
        self._actions = {value:i for i,value in enumerate(self.ACTIONS)}
        self._invActions = {i:value for i, value in enumerate(self.ACTIONS)}

        self._totalgold = totalgold
        self._players = players
        self._observation = {}

        global_space = {'GoldHostileL':spaces.Discrete(2),
                        'GoldHostileR': spaces.Discrete(2),
                        'GoldMine': spaces.Discrete(totalgold+1),
                        'GoldHome':spaces.Discrete(totalgold+1)}

        for number in range(1,players+1):
            player_space = {"Player%dPosition" % number: spaces.Discrete(len(self.POSITIONS)),'Player1HasGold':spaces.Discrete(2)}
            global_space= {**global_space,**player_space}

        # build the observation space from the global and players's state
        self.observation_space = spaces.Dict(global_space)

        # the player's turn to move
        self.active_player = 1
        # keeps track of each player state visits
        self._visited = {}
        self.seed()
        self.reset()

        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        '''
        Reset the environment
        :return: the initial observation state
        '''
        self.steps = 0
        for id in range(1,self._players+1):
            self._observation['Player%dPosition' % id] = self._positions[self.POSITIONS[0]]
            self._observation['Player%dHasGold' %id] = 0
            self._observation['GoldMine'] = self._totalgold
            self._observation['GoldHostileL'] = 0
            self._observation['GoldHostileR'] = 0
            self._observation['GoldHome'] = 0
            self._visited['Player%d' % id] = set()

        self._lastAction = None
        self._lastReward = 0.0

        return self._observation

    def _move_players(self,action_index):

        action = self.ACTIONS[action_index]

        playernumber = self.active_player
        player_position= 'Player%dPosition' % playernumber
        pos_index = self._observation[player_position]
        pos_state = self.POSITIONS[pos_index]

        has_gold = self._observation['Player%dHasGold' % playernumber]
        new_position = pos_state

        player_reward = 0.0

        # position is HOME
        if pos_state == self.POSITIONS[0]:
            # if the player is HOME and has GOLD stash it!
            if action == 'U':
                new_position = self.POSITIONS[1]

        # position is SAFE
        if pos_state == self.POSITIONS[1]:
            if action == 'D':
                # go back home
                new_position = self.POSITIONS[0]

            elif action == 'R':
                new_position = 'HostileR'

            elif action == 'L':
                new_position = 'HostileL'

        # position is Hostile on the left: you might lose your gold here
        if pos_state == 'HostileL':
            if action == 'D':
                # go back home
                new_position = 'Free'
            elif action == 'U':
                new_position = 'Gold'
            elif action == 'R':
                new_position = 'HostileR'

        # position is Hostile on the right: you might lose your gold here
        if pos_state == 'HostileR':
            if action == 'D':
                # go back home
                new_position = 'Free'
            elif action == 'U':
                new_position = 'Gold'
            elif action == 'L':
                new_position = 'HostileL'

        if pos_state == 'Gold':

            if action == 'L':
                # go back home
                new_position = 'HostileL'
            elif action == 'R':
                # go back home
                new_position = 'HostileR'

        if new_position == 'Home':
            if has_gold:
                self._observation['GoldHome']+=1
                self._observation['Player%dHasGold' % playernumber] = 0
                player_reward = +1.0

        if new_position == 'Gold':

            if self._observation['Player%dHasGold' % playernumber] == 0:
                self._observation['Player%dHasGold' % playernumber] = 1
                self._observation['GoldMine'] -= 1

                player_reward = +1.0

        if new_position =='HostileL' or new_position =='HostileR':
            # if the player is holding a gold unit
            if has_gold:
                # there is a random chance to get killed
                hit_draw = self.np_random.randint(10, size=1)
                if hit_draw > 5:
                    # get immediate negative reward
                    player_reward = -0.5
            elif self._observation['GoldHostileL'] == 1 and new_position =='HostileL':
                player_reward = +0.5
                self._observation['Player%dHasGold' % playernumber] = 1
                self._observation['GoldHostileL'] = 0
            elif self._observation['GoldHostileR'] == 1 and new_position =='HostileR':
                player_reward = +0.5
                self._observation['Player%dHasGold' % playernumber] = 1
                self._observation['GoldHostileR'] = 0


        if pos_state =='HostileL' or pos_state =='HostileR':

            #if you had gold and got hit .....
            if has_gold and self._lastReward < 0:
                # remove your gold
                self._observation['Player%dHasGold' % playernumber] = 0
                # drop the gold in the hostile state
                if pos_state == 'HostileR':
                    self._observation['GoldHostileR'] = 1

                if pos_state == 'HostileL':
                    self._observation['GoldHostileL'] = 1
                player_reward = 0.0
                new_position = 'Home'
            else:
                if pos_state =='HostileR' and self._observation['GoldHostileR']==1:
                    if has_gold == 0:
                        player_reward = +0.5
                        self._observation['Player%dHasGold' % playernumber] = 1
                elif pos_state =='HostileL' and self._observation['GoldHostileL']==1:
                    if has_gold == 0:
                        player_reward = +0.5
                        self._observation['Player%dHasGold' % playernumber] = 1

        self._observation['Player%dPosition' % playernumber] = self._positions[new_position]

        self._visited['Player%d' % playernumber].add( self._positions[new_position] )

        self._players_reward = player_reward


    def _check_terminate(self):

        # if we moved all the gold from the mine to home then it ends
        if self._observation['GoldHome'] == self._totalgold or self.steps >= self.maxsteps:
            return True
        else:
            return False

        # # TODO: too restrictive, terminate after visiting all positions
        # player_ends = 0
        # for playernumber in range(1, self._players + 1):
        #
        #     if len(self._visited['Player%d' % playernumber]) == len(self.POSITIONS):
        #         player_ends += 1
        #
        # if player_ends == self._players:
        #     return True
        # else:
        #     return False

    def _get_reward(self):
        '''
        Rewards for each player in a list indexed from 0
        :return: a list of rewards
        '''

        return self._players_reward

    def select_player(self,playerid):

        assert playerid <= self._players
        assert playerid >= 1

        self.active_player = playerid

    def _get_info(self,action,reward):
        info = {}

        for playernumber in range(1,self._players+1):
            pos_index = self._observation['Player%dPosition' % playernumber]

            info['Player%dPosition' % playernumber] = self.POSITIONS[pos_index]

            info['Player%dHasGold' % playernumber] = (True if self._observation['Player%dHasGold' % playernumber]>0 else False)


        info['ActivePlayer'] = self.active_player
        info['PreviousAction'] = '' if action is None else self._invActions[action]
        info['PlayerReward'] = reward
        info['GoldHome'] = self._observation['GoldHome']
        info['GoldHostileR'] = self._observation['GoldHostileR']
        info['GoldHostileL'] = self._observation['GoldHostileL']

        return info

    def step(self, action):

        assert self.action_space.contains(action)
        self._lastAction = action

        self._move_players(action)

        reward = self._get_reward()

        if self._check_terminate():
            done = 1
        else:
            done = 0

        info = self._get_info(action,reward)
        self._lastReward = reward

        self.steps += 1

        return self._observation, reward, done, info

    def get_last_info(self):
        info = self._get_info(self._lastAction)
        return info

    def _get_obs(self):

        return self._observation

    def _draw_rgb(self):

        gold_xy = (0.0, 3.0)
        home_xy = (0.0, -3.0)
        free_xy = (0.0, -1.0)
        hostile_lxy = (-1.0, 1.0)
        hostile_rxy = (1.0, 1.0)

        locations = [home_xy,free_xy,hostile_lxy,hostile_rxy,gold_xy]
        transitions = [(home_xy,free_xy),(free_xy,hostile_lxy),(free_xy,hostile_rxy),(hostile_rxy,gold_xy),(hostile_lxy,gold_xy),(hostile_lxy,hostile_rxy)]
        red = (1.0,0.0,0.0)
        green = (.0,1.0,.0)
        blue = (.0, .0, 1.0)
        gold = (1.0,1.0,0.0)
        colors = [blue,green,red,red,gold]
        space_radius = 0.4

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            screen_width = 400
            screen_height = 800

            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-2.0, 2.0, -4.0, 4.0)

            for i,connect in enumerate(transitions):
                line = rendering.make_line(connect[0],connect[1])
                line.set_linewidth(10)
                self.viewer.add_geom(line)

            for i,location in enumerate(locations):
                color = colors[i]
                place = rendering.make_circle(space_radius)
                place.set_color(color[0],color[1],color[2])
                place_pos = rendering.Transform(translation=location)
                place.add_attr(place_pos)
                self.viewer.add_geom(place)

        self.viewer.remove_labels()

        for id in range(1,self._players+1):
            pos_index = self._observation['Player%dPosition' % id]
            location_xy = locations[pos_index]
            # TODO: extend to any number of players
            if id == 1:
                print_xy = (location_xy[0] + 0.2,location_xy[1])
            if id == 2:
                print_xy = (location_xy[0] - 0.2, location_xy[1])

            self.viewer.draw_label(text='P1', position=print_xy)

        self.viewer.draw_label(text='Gold Home = %d' % self._observation['GoldHome'], position=(-1.2, -3.4),font_size=12)

        self.viewer.draw_label(text='Player Gold = %d' % self._observation['Player%dHasGold' % 1], position=(-1.2, -3.55),
                               font_size=12)
        if self._lastAction:
            self.viewer.draw_label(text='Player Action = %s' % self.ACTIONS[self._lastAction], position=(-1.2, -3.65),
                                   font_size=12)

        if self._lastReward:
            self.viewer.draw_label(text='Reward = %.2f' % self._lastReward, position=(-1.2, -3.9),
                                   font_size=12)

        self.viewer.draw_label(text='Gold Mine = %d' % self._observation['GoldMine'], position=(-1.2,3.5),font_size=14)
        self.viewer.draw_label(text='Step = %d' % self.steps, position=(1.2, 3.5), font_size=14)

        if self._observation['GoldHostileL']>0:
            self.viewer.draw_label(text='Gold = 1', position=(-1.5, 0.6),font_size=14)
        else:
            self.viewer.draw_label(text='Gold = 0', position=(-1.5, 0.6), font_size=14)

        if self._observation['GoldHostileR']>0:
            self.viewer.draw_label(text='Gold = 1', position=(1.5, 0.6),font_size=14)
        else:
            self.viewer.draw_label(text='Gold = 0', position=(1.5, 0.6),font_size=14)

    def render(self, mode='human'):

        if mode == 'ansi':
            outfile = StringIO()
            outfile.write(self._ascii_grid)
            return outfile
        elif mode =='human':
            outfile = sys.stdout
            outfile.write(self._ascii_grid)
        elif mode == 'rgb_array':
            if len(self._observation) == 0:
                return None
            else:
                self._draw_rgb()
                return self.viewer.render(True)

    def close(self):
        if self.viewer: self.viewer.close()