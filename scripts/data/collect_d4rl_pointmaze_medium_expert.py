# D4RL/pointmaze/medium-v2 EXPERT (Minari) -> swm HDF5. Needs `pip install
# minari`; unlike the random collector this reads Minari's own trajectories.
# States are SET and rendered rather than replayed by action: the
# dataset is one continuing rollout sliced into episodes, so
# action-replay desyncs past ~ep 100.
#     MUJOCO_GL=egl python scripts/data/collect_d4rl_pointmaze_medium_expert.py

import argparse
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import hdf5plugin
import minari
import numpy as np
from PIL import Image
from tqdm import tqdm

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Writer, _is_string_col

# Top-down view of the medium maze: the U-maze camera pulled back by 8/5.
CAMERA = {
    'distance': 14.08,
    'elevation': -90.0,
    'azimuth': 180.0,
    'lookat': np.array([0.0, 0.0, 0.0]),
}

BLOSC = hdf5plugin.Blosc(
    cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
)
DATASET_ID = 'D4RL/pointmaze/medium-v2'


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
    # Default: convert the whole dataset (4,752 episodes / 1M steps).
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--img-size', type=int, default=224)
    args = parser.parse_args()

    dataset = minari.load_dataset(DATASET_ID, download=True)
    env = dataset.recover_environment(render_mode='rgb_array')
    point_env = env.unwrapped.point_env
    point_env.mujoco_renderer.default_cam_config = CAMERA
    # Hide the goal marker so it is not rendered.
    point_env.model.site_rgba[env.unwrapped.target_site_id, 3] = 0.0
    env.reset(seed=0)  # once, to build the sim; states are set below

    n = args.episodes or dataset.total_episodes
    ids = list(range(n))

    out_path = (
        Path(swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'd4rl_pointmaze_medium_expert.h5'
    )
    writer = swm.data.get_format('hdf5').open_writer(
        out_path, mode='overwrite'
    )
    with writer as w:
        for ep in tqdm(
            dataset.iterate_episodes(ids), total=len(ids), desc='Converting'
        ):
            state = ep.observations['observation']  # (T+1, 4)
            goal = ep.observations['desired_goal']  # (T+1, 2)
            size = (args.img_size, args.img_size)
            pixels = []
            for t in range(len(state)):
                point_env.set_state(
                    state[t][:2].astype(np.float64),
                    state[t][2:].astype(np.float64),
                )
                frame = Image.fromarray(np.asarray(env.render()))
                frame = frame.resize(size, Image.BILINEAR)
                pixels.append(np.asarray(frame, dtype=np.uint8))
            nan_action = np.full(2, np.nan, dtype=np.float32)
            w.write_episode(
                {
                    'pixels': pixels,
                    'action': [a.astype(np.float32) for a in ep.actions]
                    + [nan_action],
                    'proprio': [s.astype(np.float32) for s in state],
                    # desired_goal is not rendered; kept for reference.
                    'desired_goal': [g.astype(np.float32) for g in goal],
                    # The swm writer emits only ep_len/ep_offset; eval needs
                    # per-row episode and step labels.
                    'ep_idx': [ep.id] * len(pixels),
                    'step_idx': list(range(len(pixels))),
                }
            )

    env.close()
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
