import argparse
import random
import shutil
from pathlib import Path


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def prepare(root, test_fraction, val_fraction, seed):
    raw = root / 'raw'
    if not raw.exists():
        raise FileNotFoundError(f'Create class folders inside {raw} first')
    class_dirs = sorted(p for p in raw.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError('At least two class folders are required')
    for split in ('train', 'val', 'test'):
        if (root / split).exists():
            raise FileExistsError(f'{root / split} already exists. Move or remove it before creating a new split')
    rng = random.Random(seed)
    summary = {}
    for class_dir in class_dirs:
        images = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(images) < 5:
            raise ValueError(f'{class_dir.name} needs at least five images, found {len(images)}')
        summary[class_dir.name] = len(images)
    for class_dir in class_dirs:
        images = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        rng.shuffle(images)
        test_count = max(1, round(len(images) * test_fraction))
        val_count = max(1, round(len(images) * val_fraction))
        splits = {
            'test': images[:test_count],
            'val': images[test_count:test_count + val_count],
            'train': images[test_count + val_count:]
        }
        for split, items in splits.items():
            destination = root / split / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for item in items:
                shutil.copy2(item, destination / item.name)
            print(f'{class_dir.name}: {split} = {len(items)}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or not 0 < args.val_fraction < 1 or args.test_fraction + args.val_fraction >= 1:
        parser.error('Fractions must be between zero and one, and their sum must be below one')
    prepare(Path(args.data), args.test_fraction, args.val_fraction, args.seed)
