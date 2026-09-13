# Random-policy PointMaze U-maze rollouts -> swm HDF5.
#     MUJOCO_GL=egl python scripts/data/collect_d4rl_pointmaze_u_maze_random.py

import argparse
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers PointMaze envs)
import h5py
import hdf5plugin
import numpy as np
from gymnasium_robotics.envs.maze.maps import U_MAZE
from PIL import Image
from tqdm import tqdm

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Writer, _is_string_col

# Top-down view of the U-maze, matching DINO-WM.
CAMERA = {
    'distance': 8.8,
    'elevation': -90.0,
    'azimuth': 180.0,
    'lookat': np.array([0.0, 0.0, 0.0]),
}

# Following DINO-WM, start each episode with a random velocity; from rest,
# uniform-random actions barely move the ball.
INIT_VEL = 5.2262554  # DINO-WM STATE_RANGES velocity bound
BLOSC = hdf5plugin.Blosc(
    cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
)


def init_schema(self, sample_ep: dict) -> None:
    """HDF5Writer._init_schema, copied verbatim + Blosc on the data columns.

    The swm writer creates datasets with no compression; this is the same
    method with **BLOSC added to the per-step column create_dataset so pixels
    compress on write.
    """
    for col, vals in sample_ep.items():
        if _is_string_col(vals[0]):
            self._f.create_dataset(
                col,
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(),
                chunks=(1,),
            )
            continue
        sample = np.asarray(vals[0])
        self._f.create_dataset(
            col,
            shape=(0, *sample.shape),
            maxshape=(None, *sample.shape),
            dtype=sample.dtype,
            chunks=(1, *sample.shape),
            **BLOSC,
        )
    self._f.create_dataset(
        'ep_len', shape=(0,), maxshape=(None,), dtype=np.int32
    )
    self._f.create_dataset(
        'ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64
    )


HDF5Writer._init_schema = init_schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=224)
    args = parser.parse_args()

    # Non-continuing: fresh start and goal per episode, ends on success.
    env = gym.make(
        'PointMaze_UMaze-v3',
        maze_map=U_MAZE,
        reward_type='sparse',
        continuing_task=False,
        reset_target=False,
        render_mode='rgb_array',
    )
    point_env = env.unwrapped.point_env
    point_env.mujoco_renderer.default_cam_config = CAMERA
    # Hide the goal marker so it is not rendered.
    point_env.model.site_rgba[env.unwrapped.target_site_id, 3] = 0.0

    rng = np.random.default_rng(0)

    out_path = (
        Path(swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'd4rl_pointmaze_u_maze_random.h5'
    )
    writer = swm.data.get_format('hdf5').open_writer(
        out_path, mode='overwrite'
    )
    with writer as w:
        for ep in tqdm(range(args.episodes), desc='Collecting'):
            env.reset(seed=ep)
            # Goal is fixed for the episode (reset_target=False), not rendered.
            desired_goal = np.asarray(env.unwrapped.goal, dtype=np.float32)
            qpos = point_env.data.qpos.copy()
            qvel = rng.uniform(-INIT_VEL, INIT_VEL, size=2)
            point_env.set_state(qpos, qvel)
            pixels, actions, proprio = [], [], []
            for _ in range(args.steps):
                a = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
                a = a.astype(np.float32)
                obs, _, term, trunc, _ = env.step(a)
                frame = Image.fromarray(np.asarray(env.render()))
                size = (args.img_size, args.img_size)
                frame = frame.resize(size, Image.BILINEAR)
                pixels.append(np.asarray(frame, dtype=np.uint8))
                actions.append(a)
                proprio.append(obs['observation'].astype(np.float32))
                if term or trunc:
                    break
            w.write_episode(
                {
                    'pixels': pixels,
                    'action': actions,
                    'proprio': proprio,
                    'desired_goal': [desired_goal] * len(pixels),
                    # The swm writer emits only ep_len/ep_offset; eval needs
                    # per-row episode and step labels.
                    'ep_idx': [ep] * len(pixels),
                    'step_idx': list(range(len(pixels))),
                }
            )

    env.close()
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
