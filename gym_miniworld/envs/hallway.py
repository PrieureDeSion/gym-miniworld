import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame
from gym import spaces

class Hallway(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    at the end of a hallway
    """

    def __init__(self, length=10, stochastic=False, dense_reward=True, **kwargs):
        assert length >= 5
        self.length = length
        self.stochastic = stochastic
        self.dense_reward = dense_reward

        super().__init__(
            max_episode_steps=250,
            domain_rand=True,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Create a long rectangular room
        room = self.add_rect_room(
            min_x=-1, max_x=-1 + self.length,
            min_z=-2, max_z=2
        )

        # Place the box at the end of the hallway
        if not self.stochastic:
            self.box = self.place_entity(
                Box(color='red'),
                pos=[room.max_x-2, 0, 1]
            )
        else:
            self.box = self.place_entity(
                Box(color='red'),
                min_x=room.max_x - 1
            )
        self.entities.append(ImageFrame(
            pos=[1, 1.35, 2],
            dir=math.pi/2,
            width=2.5,
            tex_name='portraits/viktor_vasnetsov'
        ))

        self.entities.append(ImageFrame(
            pos=[1 + self.length / 2, 1.55, 2],
            dir=math.pi/2,
            width=2.5,
            tex_name='portraits/robert_dampier'
        ))

        self.entities.append(ImageFrame(
            pos=[1, 1.35, -2],
            dir=-math.pi/2,
            width=2.5,
            tex_name='portraits/nathaniel_jocelyn'
        ))

        self.entities.append(ImageFrame(
            pos=[1 + self.length / 2, 1.45, -2],
            dir=-math.pi/2,
            width=2.5,
            tex_name='portraits/robert_leopold'
        ))

        self.entities.append(ImageFrame(
            pos=[-1+self.length, 1.35, 0],
            dir=math.pi,
            width=2,
            tex_name='chars/ch_5'
        ))

        self.entities.append(ImageFrame(
            pos=[-1, 1.35, 0],
            dir=0,
            width=2,
            tex_name='chars/ch_0'
        ))

        # Place the agent a random distance away from the goal
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            max_x=room.max_x - 3
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # note that reward is always 0 here
        
        if self.dense_reward:
            dist = np.linalg.norm(self.agent.pos - self.box.pos)
            max_dist = np.linalg.norm([self.min_z-self.max_z, self.max_x-self.min_x])
            reward = -1 * dist / max_dist / self.max_episode_steps
            
            if self.near(self.box):
                reward = 1
                done = True
        else:
            if self.near(self.box):
                reward += self._reward()
                done = True

        return obs, reward, done, info

