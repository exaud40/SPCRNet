
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class FramesDataset(Dataset):
    def __init__(self, rows: list[dict], img_root: str | Path, train: bool = False, seed: int = 0,
                 grids: dict | None = None, grid_shape: tuple[int, int] = (9, 16)):
        self.rows, self.root, self.train = rows, Path(img_root), train
        self.rng = np.random.default_rng(seed)
        self.grids, self.grid_shape = grids, grid_shape

    def __len__(self) -> int:
        return len(self.rows)

    def load(self, i: int) -> np.ndarray:
        from PIL import Image

        with Image.open(self.root / self.rows[i]["image"]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0      # (H, W, 3)

    def __getitem__(self, i: int):
        x = self.load(i)
        r = self.rows[i]
        g = np.zeros(self.grid_shape, np.float32)
        if self.grids is not None:
            g = self.grids.get((r["seq"], r["view"], r["frame"]), g).astype(np.float32)
        if self.train:
            rng = np.random.default_rng((self.rng.integers(1 << 31) + i) % (1 << 31))
            if rng.random() < 0.5:
                x, g = x[:, ::-1], g[:, ::-1]
            b, c = rng.uniform(-0.15, 0.15), rng.uniform(0.8, 1.2)
            x = np.clip((x - 0.5) * c + 0.5 + b, 0.0, 1.0)
        x = (x - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), int(r["label"]), i,
                torch.from_numpy(np.ascontiguousarray(g)), float(r["n_union"]))
