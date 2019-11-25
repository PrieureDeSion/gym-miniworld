from gym_miniworld.miniworld import MiniWorldEnv, Room
from gym_miniworld.entity import Box
from repnav.dependencies import *


def get_gt_state_rep(environment, agent=None):
    """
    get ground truth state representation from TFenv
    """
    if agent is None:
        env = environment.pyenv.envs[0]
        while True:
            if not isinstance(env, gym_miniworld.miniworld.MiniWorldEnv):
                env = env.env
                continue
            break
        agent = env.agent
    position = agent.pos
    heading = agent.dir
    return np.append(np.append(position, np.sin(heading)), np.cos(heading))

class HallwayGoal(MiniWorldEnv):
    """
    Environment with programmable goal setting
    Red box denotes goal (acceptable for GT state rep)
    """

    def __init__(self, length=6, stochastic=False, dense_reward=True, **kwargs):
        assert length >= 5
        self.length = length
        self.stochastic = stochastic
        self.dense_reward = dense_reward

        super().__init__(
            max_episode_steps=100,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        # self.action_space = spaces.Discrete(self.actions.move_forward+1)
        self.action_space = spaces.Box(low=0., high=2., shape=(1,), dtype=np.float32)

    def _gen_world(self):
        # Create a long rectangular room
        room = self.add_rect_room(
            min_x=-1, max_x=-1 + self.length,
            min_z=-2, max_z=2
        )

        # Place the agent a random distance away from the goal
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            max_x=room.max_x
        )
        
        self.goal = self.place_entity(
                Box(color='red'))

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
    
    def get_agent_pos(self, agent=None):
        if agent is None:
            agent = self.agent
        return agent.pos
    
    def get_agent_dir(self, agent=None):
        if agent is None:
            agent = self.agent
        return [np.sin(agent.dir), np.cos(agent.dir)]
    
    def get_agent_state(self, agent=None):
        if agent is None:
            agent = self.agent
        return np.concatenate([self.get_agent_pos(agent), self.get_agent_dir(agent)])


def visualize_rollout(environment, policy):
    """
    Visualize a rollout of the policy in the environment
    """
    episode_return = 0.0
    time_step = environment.reset()
    env = environment.pyenv.envs[0].env
    img = plt.imshow(np.squeeze(env.render_obs()))
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        img.set_data(np.squeeze(env.render_obs()))
        display.display(plt.gcf())
        display.clear_output(wait=True)
        episode_return += time_step.reward
        
class GoalConditionedMiniWorldWrapper(gym.Wrapper):
    """Wrapper that appends goal to state produced by environment."""

    def __init__(self, env, prob_constraint=0, min_dist=0, max_dist=4,
                 threshold_distance=1.0):
        """Initialize the environment.

        Args:
          env: an environment.
          prob_constraint: (float) Probability that the distance constraint is
            followed after resetting.
          min_dist: (float) When the constraint is enforced, ensure the goal is at
            least this far from the initial state.
          
          edist: (float) When the constraint is enforced, ensure the goal is at
            most this far from the initial state.
          threshold_distance: (float) States are considered equivalent if they are
            at most this far away from one another.
        """
        self._threshold_distance = threshold_distance
        self._prob_constraint = prob_constraint
        self._min_dist = min_dist
        self._max_dist = max_dist
        super(GoalConditionedMiniWorldWrapper, self).__init__(env)
        
        self.wrapped_env = self.env
        # ensure self.env is a subclass of MiniWorld and not a wrapper
        while True:
            if not isinstance(self.env, gym_miniworld.miniworld.MiniWorldEnv):
                self.env = self.env.env
                continue
            break
        
        self.observation_space = gym.spaces.Dict({
            'observation': env.observation_space,
            'goal': env.observation_space,
        })

    def _normalize_obs(self, obs):
        if len(obs) == 5:
            # state obs; sin/cos of heading is already normalized
            return np.array([
                obs[0] / float(self.env.length),
                0.,
                obs[2] / float(self.env.length),
                obs[3],
                obs[4]
            ])
        else:
            print(obs)
            raise Exception('Invalid observation for normalization')
            
    def _unnormalize_obs(self, obs):
        if len(obs) == 5:
            # state obs; sin/cos of heading is already normalized
            return np.array([
                obs[0] * float(self.env.length),
                0.,
                obs[2] * float(self.env.length),
                obs[3],
                obs[4]
            ])
        else:
            print(obs)
            raise Exception('Invalid observation for normalization')


    def reset(self):
        goal = None
        while goal is None:
            obs = self.wrapped_env.reset()
            goal = self.sample_goal()

        self._goal = get_gt_state_rep(self.env, self.env.goal)
        return {'observation': self._normalize_obs(obs),
                'goal': self._normalize_obs(self._goal)}

    def step(self, action):
        obs, rew, done, info = self.wrapped_env.step(action)
        return {'observation': self._normalize_obs(obs),
                'goal': self._normalize_obs(self._goal)}, rew, done, {}

    def set_sample_goal_args(self, prob_constraint=0.75,
                             min_dist=1, max_dist=7):
        assert prob_constraint is not None
        assert min_dist is not None
        assert max_dist is not None
        assert min_dist >= 0
        assert max_dist >= min_dist
        self._prob_constraint = prob_constraint
        self._min_dist = min_dist
        self._max_dist = max_dist


    def set_sample_goal_args(self, prob_constraint=None,
                               min_dist=None, max_dist=None):
        assert prob_constraint is not None
        assert min_dist is not None
        assert max_dist is not None
        assert min_dist >= 0
        assert max_dist >= min_dist
        self._prob_constraint = prob_constraint
        self._min_dist = min_dist
        self._max_dist = max_dist
    
    def sample_goal(self):
        """
        Sample a goal state
        """
        if np.random.random() < self._prob_constraint:
            return self.sample_goal_constrained(self._min_dist, self._max_dist)
        else:
            return self.sample_goal_unconstrained()
        
        
    def sample_goal_constrained(self, min_dist, max_dist):
        """
        sample a random (valid) goal in the same room as agent and distance
        bounded by [min_dist, max_dist].
        Since a room is bounded and convex (rectangle), this considers the shortest
        feasible path without obstacles
        """
        
        # restrict goal to agent's room
        r = None
        for room in self.env.rooms:
            if room.point_inside(self.env.agent.pos):
                r = room
                break
        assert r is not None
        # Choose a random point within the square bounding box of the room
        lx = r.min_x
        hx = r.max_x
        lz = r.min_z
        hz = r.max_z
        
        count = 0
        while True:
            count += 1
            if (count > 100):
                print("WARNING: Unable to find goal within constraints")
                
            pos = self.env.rand.float(
                low =[lx + self.env.goal.radius, 0, lz + self.env.goal.radius],
                high=[hx - self.env.goal.radius, 0, hz - self.env.goal.radius]
            )

            # Make sure the position is within the room's outline
            if not r.point_inside(pos):
                continue

            # Make sure the position doesn't intersect with any walls
            if self.env.intersect(self.env.goal, pos, self.env.goal.radius):
                continue
                
            agent_dist = np.linalg.norm(self.env.goal.pos - self.env.agent.pos)
            if agent_dist > max_dist or agent_dist < min_dist:
                continue
            
            self.env.goal.pos = pos
            self.env.goal.dir = 0 # goal is heading-agnostic
            break
        
        return np.concatenate([pos, [0., 1.]])
        
        
    def sample_goal_unconstrained(self, min_dist, max_dist):
        """
        sample a random (valid) goal in the environment and place a red box
        """
        
        count = 0
        while True:
            count += 1
            if (count > 100):
                print("WARNING: Unable to find goal within constraints")
            if (count > 1000):
                print("WARNING: Unable to find a goal; sampling constrained")
                
            # Pick a room, sample rooms proportionally to floor surface area
            r = self.env.rand.choice(self.env.rooms, probs=self.env.room_probs)
            
            # Choose a random point within the square bounding box of the room
            lx = r.min_x
            hx = r.max_x
            lz = r.min_z
            hz = r.max_z
            pos = self.env.rand.float(
                low =[lx + self.env.goal.radius, 0, lz + self.env.goal.radius],
                high=[hx - self.env.goal.radius, 0, hz - self.env.goal.radius]
            )

            # Make sure the position is within the room's outline
            if not r.point_inside(pos):
                continue

            # Make sure the position doesn't intersect with any walls
            if self.env.intersect(self.env.goal, pos, self.env.goal.radius):
                continue

            agent_dist = np.linalg.norm(self.env.goal.pos - self.env.agent.pos)
            if agent_dist > max_dist or agent_dist < min_dist:
                continue

            self.env.goal.pos = pos
            self.env.goal.dir = 0 # goal is heading-agnostic
            break
        
        return np.concatenate([pos, [0., 1.]])