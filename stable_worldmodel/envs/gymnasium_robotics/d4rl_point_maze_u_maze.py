"""PointMaze U-maze env."""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers PointMaze envs)
import numpy as np
from gymnasium_robotics.envs.maze.maps import U_MAZE
from PIL import Image

from stable_worldmodel import spaces as swm_spaces

# Top-down view of the U-maze (matching DINO-WM and the collectors exactly).
CAMERA = {
    'distance': 8.8,
    'elevation': -90.0,
    'azimuth': 180.0,
    'lookat': np.array([0.0, 0.0, 0.0]),
}


class D4RLPointMazeUMazeEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    # DINO-WM's success threshold (point_maze_wrapper.eval_state).
    SUCCESS_RADIUS = 0.5

    def __init__(self, render_mode='rgb_array', img_size=224, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.img_size = img_size
        self.env_name = 'D4RLPointMazeUMaze'
        self.image_shape = (img_size, img_size, 3)
        self.state_dim = 4

        self.env = gym.make(
            'PointMaze_UMaze-v3',
            maze_map=U_MAZE,
            reward_type='sparse',
            continuing_task=False,
            reset_target=False,
            render_mode='rgb_array',
        )
        self.maze = self.env.unwrapped
        self.point_env = self.maze.point_env
        self.point_env.mujoco_renderer.default_cam_config = CAMERA
        # Hide the red target site (DINO-WM renders no goal marker).
        self.point_env.model.site_rgba[self.maze.target_site_id, 3] = 0.0

        # 2-D force in [-1, 1]; state (x, y, vx, vy).
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Low-dim state; the frame is delivered via render() (swm's
        # AddPixelsWrapper writes it as `pixels`).
        self.observation_space = swm_spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.variation_space = swm_spaces.Dict({})

        self.last_image = np.zeros(self.image_shape, dtype=np.uint8)
        self.goal_xy = None  # set by set_goal during eval

    def proprio(self):
        qpos = self.point_env.data.qpos[:2].astype(np.float32)
        qvel = self.point_env.data.qvel[:2].astype(np.float32)
        return np.concatenate([qpos, qvel])

    def render_current(self):
        frame = Image.fromarray(np.asarray(self.env.render()))
        frame = frame.resize((self.img_size, self.img_size), Image.BILINEAR)
        self.last_image = np.asarray(frame, dtype=np.uint8)
        return self.last_image

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=int(seed) if seed is not None else 0)
        options = options or {}
        state = options.get('state')
        if state is not None:
            self.set_state(state)
        else:
            self.render_current()
        proprio = self.proprio()
        return proprio, self.make_info(proprio)

    def step(self, action):
        action = np.nan_to_num(np.asarray(action, dtype=np.float32))
        self.env.step(action)
        self.render_current()
        proprio = self.proprio()
        return (
            proprio,
            0.0,
            self.reached_goal(proprio),
            False,
            self.make_info(proprio),
        )

    def reached_goal(self, state):
        """DINO-WM's success test: ``||goal_xy - state_xy|| < 0.5``. False until a goal is set."""
        if self.goal_xy is None:
            return False
        return bool(
            np.linalg.norm(self.goal_xy - state[:2]) < self.SUCCESS_RADIUS
        )

    # --- eval hooks, called by the dataset-driven eval callables ---

    def set_state(self, state, desired_goal=None):
        """Set the sim to a dataset state (x, y, vx, vy) and render it."""
        if desired_goal is not None:
            self.maze.goal = np.asarray(desired_goal, dtype=np.float64)[:2]
            self.maze.update_target_site_pos()
        state = np.asarray(state, dtype=np.float64)
        self.point_env.set_state(state[:2], state[2:])
        self.render_current()

    def set_goal(self, goal):
        """Record the success target (x, y) — the state the episode must reach."""
        self.goal_xy = np.asarray(goal, dtype=np.float32)[:2]

    def render(self):
        return self.last_image

    def make_info(self, proprio):
        return {
            'env_name': self.env_name,
            'proprio': proprio,
            'state': proprio,
        }

    def close(self):
        self.env.close()
