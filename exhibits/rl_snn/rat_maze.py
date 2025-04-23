from copy import deepcopy
import sys
import numpy as np
from jax import numpy as jnp, random, jit, nn
import math

tiles = { ## base tile types for constructing sensory representations
    -1: jnp.zeros((3, 3)),
    0: jnp.array(
        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 0.]]
    ),
    1: jnp.array(
        [[1., 1., 1.],
         [1., 1., 1.],
         [0., 1., 1.]]
    ),
    2: jnp.array(
        [[1., 1., 0.],
         [1., 1., 1.],
         [1., 1., 1.]]
    ),
    3: jnp.array(
        [[0., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]]
    ),
    4: jnp.array(
        [[1., 1., 1.],
         [1., 1., 1.],
         [0., 0., 0.]]
    ),
    5: jnp.array(
        [[0., 0., 0.],
         [1., 1., 1.],
         [1., 1., 1.]]
    ),
    6: jnp.array(
        [[1., 1., 0.],
         [1., 1., 0.],
         [1., 1., 0.]]
    ),
    7: jnp.array(
        [[0., 1., 1.],
         [0., 1., 1.],
         [0., 1., 1.]]
    ),
    8: jnp.array(
        [[0., 0., 0.],
         [1., 1., 0.],
         [1., 1., 0.]]
    ),
    9: jnp.array(
        [[0., 0., 0],
         [0., 1., 1.],
         [0., 1., 1.]]
    ),
    10: jnp.array(
        [[0., 1., 1.],
         [0., 1., 1.],
         [0., 0., 0.]]
    ),
    11: jnp.array(
        [[1., 1., 0],
         [1., 1., 0.],
         [0., 0., 0.]]
    ),
    12: jnp.array( ## agent
        [[0., 1., 0],
         [1., 1., 1.],
         [0., 1., 0.]]
    ),
    13: jnp.array( ## goal
        [[1., 0., 1],
         [0., 1., 0.],
         [1., 0., 1.]]
    )
}

class RatMaze():
    """
    Construct a simulator for rat maze navigation problems (such as the rat T-maze problem).

    Args:
        dkey: JAX seeding key

        width_x: the number of tiles/discrete states along width axis (DEFAULT: 3)

        width_y: the number of tiles/discrete states along length/height axis (DEFAULT: 4)

        reward_type: string indicating type of reward functional to use (DEFAULT: "dist")

        maze_type: string indicating type of maze environment to simulate  (DEFAULT: "t_maze")

        is_deterministic: <unused> (DEFAULT: True)

        episode_len: number of steps to simulate before an episode's termination, i.e., length of a single episode (DEFAULT: 30)

        sensory_encoding: string indicating type of sensory input encoding to produce (DEFAULT: "pixels")

            :Note: Options include "pixels" (for pixel images) and "coordinates" (for one-hot encoded x-y coordinates of the agent)
    """

    def __init__(
            self, dkey, width_x=3, width_y=4, reward_type="dist", maze_type="t_maze", is_deterministic=True,
            episode_len=30, sensory_encoding="pixels", **kwargs
    ):
        #dkey, *subkeys = random.split(dkey, 15)
        self.dkey = dkey
        self.width_x = width_x
        self.width_y = width_y
        self.maze_type = maze_type.lower()
        self.reward_type = reward_type ## "dist"  "dist_decayed"  "sparse"

        self.true_reward = -1000.
        self.is_deterministic = is_deterministic

        self._shifts = [ ## possible shifts in cardinal directions
            jnp.array([1, 0]), jnp.array([-1, 0]), jnp.array([0, 1]), jnp.array([0, -1])
        ]

        ################################################################################################################
        ## internal world representation (one integer per tile)
        if self.maze_type == "t_maze": ## rat t-maze problem
            midpoint = int(math.ceil(self.width_x / 2))
            self._agent_pos = jnp.array([self.width_y, midpoint])
            self._init_agent_pos = self._agent_pos + 0
            self._goal_pos = jnp.array([1, self.width_x])

            midpoint = int(math.ceil(self.width_x / 2))
            self.mini_world = jnp.zeros((self.width_y + 2, self.width_x + 2), dtype=jnp.int32) - 1  ## (y, x)
            self.mini_world = self.mini_world.at[0, 0].set(0)
            self.mini_world = self.mini_world.at[1, 0].set(6)
            self.mini_world = self.mini_world.at[2, 0].set(2)
            self.mini_world = self.mini_world.at[0, self.width_x + 1].set(1)
            self.mini_world = self.mini_world.at[1, self.width_x + 1].set(7)
            self.mini_world = self.mini_world.at[2, self.width_x + 1].set(3)
            for i in range(1, self.width_x + 1):
                self.mini_world = self.mini_world.at[0, i].set(4)  ## top row
                if midpoint - 1 < i < midpoint + 1:
                    self.mini_world = self.mini_world.at[width_y + 1, i].set(5)  ## bot row
                elif i != midpoint:
                    self.mini_world = self.mini_world.at[2, i].set(5)  ## bot row
                if i == midpoint - 1:
                    self.mini_world = self.mini_world.at[2, i].set(8)
                    for j in range(3, self.width_y + 1):
                        self.mini_world = self.mini_world.at[j, i].set(6)
                if i == midpoint + 1:
                    self.mini_world = self.mini_world.at[2, i].set(9)
                    for j in range(3, self.width_y + 1):
                        self.mini_world = self.mini_world.at[j, i].set(7)
            self.mini_world = self.mini_world.at[width_y + 1, midpoint - 1].set(2)
            self.mini_world = self.mini_world.at[width_y + 1, midpoint + 1].set(3)
            self.collision_field = (self.mini_world >= 0) * 1.  ## for collision detection
        else:
            print(f"ERROR: Unsupported maze environment type {maze_type}")
            sys.exit(1)
        ################################################################################################################

        self.start_pos = None
        self.closest_pos = self._agent_pos + 0
        self.n_steps_taken = 0
        self.max_ep_len = episode_len
        self.dist_best = -1000.

        ## construct full tiled world/environment
        _world = []
        for i in range(self.mini_world.shape[0]):
            row = []
            for j in range(self.mini_world.shape[1]):
                v = int(self.mini_world[i, j])
                tile_v = tiles.get(v)
                row.append(tile_v)
            #row = jnp.concatenate(row, axis=1)
            _world.append(row)
        self._world = _world #jnp.concatenate(_world, axis=0)
        board = self._build_world_rep((0,0), (0,0))
        self.dim_x = board.shape[0]
        self.dim_y = board.shape[1]

    def get_dim(self): ## get dimensionality of raw sensory space
        return self.mini_world.shape[0] * self.mini_world.shape[1] * 3 * 3

    def _build_world_rep(self, agent_xy, goal_xy): ## internal co-routine to construct internal world rep
        x, y = agent_xy
        gx, gy = goal_xy
        _world = deepcopy(self._world)
        _world[x][y] = tiles.get(12) ## set agent
        _world[gx][gy] = tiles.get(13) ## set goal
        tmp = []
        for i in range(len(_world)):
            row = []
            for j in range(len(_world[i])):
                row.append(_world[i][j])
            tmp.append(jnp.concatenate(row, axis=1))
        _world = jnp.concatenate(tmp, axis=0)
        return _world

    def render(self, raw_env_pixels=False):
        ax = self._agent_pos[0]
        ay = self._agent_pos[1]
        if raw_env_pixels: ## render full scale environment
            smear_factor = 15 # 9
            maze = self.encode(jnp.array([ax, ay]), get_world=True, flatten=False)
            maze = jnp.repeat(maze, repeats=smear_factor, axis=0)
            maze = jnp.repeat(maze, repeats=smear_factor, axis=1) * 255. ## scale back to pixel space
        else: ## render just a single element/integer-coded representation of environment
            gx = self._goal_pos[0]
            gy = self._goal_pos[1]
            if ax == gx and ay == gy: ## reached goal state (mark it!)
                maze = self.collision_field.at[gx, gy].set(4) # 4 marks the spot
            else: ## else, render agent and goal position
                _world = self.collision_field.at[gx, gy].set(3)
                maze = _world.at[ax, ay].set(2)
                maze = np.asarray(maze)
        return maze

    def encode(self, coords, get_world=True, flatten=True): ## obtain encoding of current state
        x =  jnp.squeeze(coords)[0]
        y =  jnp.squeeze(coords)[1]
        ex = jnp.expand_dims(nn.one_hot(x, num_classes=self.width_x), axis=0)
        ey = jnp.expand_dims(nn.one_hot(y, num_classes=self.width_y), axis=0)
        coords = jnp.concatenate([ex, ey], axis=1)
        if get_world:
            gx = self._goal_pos[0]
            gy = self._goal_pos[1]
            world = self._build_world_rep(agent_xy=(x, y), goal_xy=(gx, gy))
            if flatten:
                world = jnp.reshape(world, (1, world.shape[0] * world.shape[1]))
            return world
        return coords

    def step(self, action): ## transition environment simulation one single step forward
        self.n_steps_taken += 1
        done = False
        reward = 0.
        _agent_pos = self._agent_pos + self._shifts[action]

        ## check boundaries
        x = _agent_pos[0]
        y = _agent_pos[1]
        collision_detect = self.collision_field[x, y]
        if collision_detect == 0:
            self._agent_pos = _agent_pos ## accept new position proposal if within legal spatial boundaries
        ## else, reject new proposal of position (out-of-bounds) -- boundary_penalty = -1.

        dist_t = jnp.linalg.norm(self._goal_pos - self._agent_pos, ord=1)  ## current distance to goal
        self.true_reward = -dist_t

        if "dist" in self.reward_type:
            decay_term = 1.
            if "decayed" in self.reward_type:
                shift = -50.
                decay_term = jnp.exp(-(self.n_steps_taken * 1.)/(self.max_ep_len - shift))
            if dist_t < self.dist_best: ## check acceleration
                reward = 1. #* decay_term
                #self.dist_best = dist_t
                reward = reward * decay_term
            else:
                reward = -1.
            self.dist_best = dist_t ## store last distance
            if dist_t < 1e-5:
                reward = 1. #* decay_term
                done = True
        else: ## reward_type == "sparse"
            reward = -0.1
            if dist_t < 1e-5:
                reward = 1.
                done = True
        state_t = self._agent_pos
        return state_t, reward, done, {} ## format: obs(t), reward, done, {information}

    def reset(self): ## reset environment back to initial conditions
        self.dist_best = jnp.linalg.norm(self._goal_pos - self._agent_pos, ord=1)
        self.true_reward = -1000.
        self._agent_pos = self._init_agent_pos + 0
        self.start_pos = self._agent_pos + 0
        self.closest_pos = self._agent_pos + 0
        self.n_steps_taken = 0
        return self._agent_pos
