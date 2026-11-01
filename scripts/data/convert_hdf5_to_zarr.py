"""Convert PetalDex HDF5 episodes to the training Zarr schema."""
import argparse
from pathlib import Path
import shutil
import tempfile
import cv2
import h5py
import numpy as np
import zarr
from numcodecs import Blosc


def _write_episodes(files, output):
    root = zarr.open_group(str(output), mode='w')
    data = root.create_group('data')
    ends = []
    total = 0
    codec = Blosc(cname='lz4', clevel=5)
    for path in files:
        with h5py.File(path, 'r') as f:
            state = np.asarray(f['observations/state'], dtype=np.float32)
            action = np.asarray(f['action'], dtype=np.float32)
            n = len(state)
            if n == 0 or state.shape != (n,54) or action.shape != (n,54):
                raise ValueError(f'{path}: expected nonempty (T,54) state and action')
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f'{path}: non-finite joint values')
            arrays = {'action': action}
            for key,a,b in [('left_arm_states',0,7),('right_arm_states',7,14),('left_hand_states',14,34),('right_hand_states',34,54)]:
                arrays[key] = state[:,a:b]
            for key, values in arrays.items():
                if key not in data:
                    data.create_dataset(key, shape=(0,*values.shape[1:]), chunks=(256,*values.shape[1:]), dtype='f4', compressor=codec)
                data[key].append(values)
            for camera,key in [('head','cam_head_color'),('right_wrist','cam_right_wrist_color')]:
                images = f[f'observations/images/{camera}']
                if len(images) != n:
                    raise ValueError(f'{path}: {camera} frame count mismatch')
                if key not in data:
                    data.create_dataset(key, shape=(0,224,224,3), chunks=(1,224,224,3), dtype='u1', compressor=codec)
                data[key].resize((total+n,224,224,3))
                for i in range(n):
                    image = cv2.imdecode(np.asarray(images[i], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape != (224,224,3):
                        raise ValueError(f'{path}: invalid {camera} JPEG at frame {i}')
                    data[key][total+i] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            total += n
            ends.append(total)
        print(f'Converted {path.name}: {n} frames', flush=True)
    root.create_group('meta').create_dataset('episode_ends', data=np.asarray(ends,dtype=np.int64))
    print(f'Saved {len(ends)} episodes, {total} frames to {output}')


def convert(source, output):
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    files = sorted(source_path.glob('episode_*.h5'))
    if not files:
        raise ValueError(f'No episode_*.h5 files in {source_path}')
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(tempfile.mkdtemp(
        prefix=f'.{output_path.name}.', dir=output_path.parent
    ))
    try:
        _write_episodes(files, temporary_path)
        temporary_path.rename(output_path)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    print(f'Completed dataset: {output_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Directory containing episode_*.h5')
    parser.add_argument('--output', required=True, help='New Zarr directory')
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == '__main__':
    main()
