from collections import defaultdict
from pathlib import Path
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LegoImageDataset(Dataset):
    def __init__(
        self,
        samples,
        class_to_idx,
        idx_to_part_num,
        idx_to_name,
        image_size=128,
        grayscale=False,
        train=False,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_part_num = idx_to_part_num
        self.idx_to_name = idx_to_name
        self.image_size = image_size
        self.grayscale = grayscale
        self.train = train

        transform_steps = [transforms.Resize((image_size, image_size))]

        if grayscale:
            transform_steps.append(transforms.Grayscale(num_output_channels=1))

        if train:
            transform_steps.extend([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])

        transform_steps.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_steps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, part_num = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[part_num]
        return image, label


class DataPreparation:
    def __init__(
        self,
        images_root,
        parts_csv_path,
        image_size=128,
        train_split=0.8,
        random_seed=42,
        grayscale=False,
        selected_parts=None,
        duplicate_singletons_for_test=True,
    ):
        self.images_root = Path(images_root)
        self.parts_csv_path = Path(parts_csv_path)
        self.image_size = image_size
        self.train_split = train_split
        self.random_seed = random_seed
        self.grayscale = grayscale
        self.selected_parts = {str(x) for x in selected_parts} if selected_parts else None
        self.duplicate_singletons_for_test = duplicate_singletons_for_test

        self.width = image_size
        self.height = image_size
        self.image_depth = 1 if grayscale else 3

        self.parts_df = pd.read_csv(self.parts_csv_path, dtype={"part_num": str})
        required_columns = {"part_num", "name"}
        missing = required_columns - set(self.parts_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in parts.csv: {sorted(missing)}")

        self.parts_df["part_num"] = self.parts_df["part_num"].astype(str).str.strip()
        self.parts_df["name"] = self.parts_df["name"].astype(str).str.strip()

        self.part_num_to_name = dict(zip(self.parts_df["part_num"], self.parts_df["name"]))

        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(
                f"No images matched parts.csv under {self.images_root}. "
                "Check your paths and extracted Kaggle dataset."
            )

        unique_part_nums = sorted({part_num for _, part_num in self.samples})
        self.classes = unique_part_nums
        self.class_to_idx = {part_num: idx for idx, part_num in enumerate(self.classes)}
        self.idx_to_part_num = {idx: part_num for part_num, idx in self.class_to_idx.items()}
        self.idx_to_name = {
            idx: self.part_num_to_name[part_num] for idx, part_num in self.idx_to_part_num.items()
        }

    def _infer_part_num(self, image_path: Path):
        candidates = [
            image_path.stem.strip(),
            image_path.parent.name.strip(),
            image_path.stem.split("_")[0].strip(),
            image_path.stem.split("-")[0].strip(),
        ]

        for candidate in candidates:
            if candidate in self.part_num_to_name:
                return candidate

        return None

    def _collect_samples(self):
        allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        samples = []

        for image_path in self.images_root.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in allowed_exts:
                continue

            part_num = self._infer_part_num(image_path)
            if part_num is None:
                continue

            if self.selected_parts and part_num not in self.selected_parts:
                continue

            samples.append((str(image_path), part_num))

        return samples

    def get_data(self):
        rng = random.Random(self.random_seed)
        grouped = defaultdict(list)

        for sample in self.samples:
            grouped[sample[1]].append(sample)

        train_samples = []
        test_samples = []

        for part_num, part_samples in grouped.items():
            local_samples = part_samples[:]
            rng.shuffle(local_samples)

            if len(local_samples) == 1:
                # This keeps the pipeline runnable for reference-style datasets.
                train_samples.append(local_samples[0])
                if self.duplicate_singletons_for_test:
                    test_samples.append(local_samples[0])
                continue

            split_idx = int(len(local_samples) * self.train_split)
            split_idx = max(1, split_idx)
            split_idx = min(split_idx, len(local_samples) - 1)

            train_samples.extend(local_samples[:split_idx])
            test_samples.extend(local_samples[split_idx:])

        train_data = LegoImageDataset(
            samples=train_samples,
            class_to_idx=self.class_to_idx,
            idx_to_part_num=self.idx_to_part_num,
            idx_to_name=self.idx_to_name,
            image_size=self.image_size,
            grayscale=self.grayscale,
            train=True,
        )

        test_data = LegoImageDataset(
            samples=test_samples,
            class_to_idx=self.class_to_idx,
            idx_to_part_num=self.idx_to_part_num,
            idx_to_name=self.idx_to_name,
            image_size=self.image_size,
            grayscale=self.grayscale,
            train=False,
        )

        return train_data, test_data

    def get_part_num_from_index(self, idx):
        return self.idx_to_part_num[idx]

    def get_part_name_from_index(self, idx):
        return self.idx_to_name[idx]