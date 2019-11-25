import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

class FourRoomsGoal(MiniWorldEnv):
    """
    Classic four rooms environment.
    The agent must reach the red box to get a reward.
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=250,
            **kwargs
        )

        # Allow only the movement actions
        # self.action_space = spaces.Discrete(self.actions.move_forward+1)
        self.action_space = spaces.Box(low=0., high=2., shape=(1,), dtype=np.float32)

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1 , max_z=7
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7
        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1 , max_x=7,
            min_z=-7, max_z=-1
        )
        # Bottom-left room
        room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        self.goal = self.place_entity(Box(color='red'))

        self.place_agent()

        min_pos = np.inf
        max_pos = -np.inf
        for r in self.rooms:
            min_pos = min(min_pos, r.min_x, r.min_z)
            max_pos = max(max_pos, r.max_x, r.max_z)

        self.length = np.abs(max_pos - min_pos)

    def step(self, action):
        if type(action) is float:
            action = int(round(action))
        elif type(action) is np.ndarray:
            action = np.asarray(np.ndarray.round(action), dtype=np.int64)

        obs, reward, done, info = super().step(action)
        # note that reward is always 0 here
        
        if self.dense_reward:
            dist = np.linalg.norm(self.agent.pos - self.goal.pos)
            max_dist = np.linalg.norm([self.min_z-self.max_z, self.max_x-self.min_x])
            reward = -1 * dist / max_dist / self.max_episode_steps
            
            if self.near(self.goal):
                reward = 1
                done = True
        else:
            if self.near(self.goal):
                reward += 1 # self._reward()
                done = True

        return obs, reward, done, info
