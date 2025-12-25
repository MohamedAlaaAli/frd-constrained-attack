#!/usr/bin/env python3
"""
experiment_frd_attack_fast.py

Faster, vectorized version of the FRD-constrained MOPSO experiment.
- Uses torch on GPU for DCT basis, image synthesis, and embedder evaluation.
- Batch-evaluates the whole swarm per iteration for speed.
- Attempts to batch-call frd_fn; falls back to per-sample if needed.
- Produces convergence CSVs, pareto front plot, trade-off plot, and sample images.

Usage:
    python experiment_frd_attack_fast.py
"""

import os
import json
import time
import math
import csv
import random
import numpy as np
from tqdm import trange
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


from frd.compute_frd import get_distance_fn
from models.clip import CLIPImageEmbedder

# -----------------------
# Config (tune to your hardware & budget)
# -----------------------
CFG = {
    "dataset_path": "test_sets/test_images/dmf",
    "output_dir": "results/dmf/pso",
    "image_size": (224, 224),
    "num_samples": 243,  # number of images to attack (set <= dataset size)
    "swarm_size": 50,
    "iters": 80,
    "d": 256,
    "alpha_scale": 0.04,
    "eps": 0.05,
    "archive_size": 50,
    "high_freq_basis": False,
    "frd_repeats": 1,
    "tau": 0.03,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "save_every": 20,  # save pareto images every N iterations
    "progress_csv_name": "progress.csv",
}

def get_args():
    parser = argparse.ArgumentParser(description="FRD-constrained MOPSO Attack")
    parser.add_argument("--dataset", type=str, default="dmf", choices=["dmf", "ham", "derm7pt"], help="Dataset to attack")
    parser.add_argument("--num_samples", type=int, default=243, help="Number of samples to attack")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--swarm_size", type=int, default=50, help="Swarm size")
    parser.add_argument("--iters", type=int, default=80, help="Number of iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

device = torch.device(CFG["device"])


TEST_IDS_PATH = "test_sets/test_ids"
TEST_IMAGES_PATH = "test_sets/test_images"

HAM_DATASET = "datasets/HAM"
DERM7PT_DATASET = "datasets/DERM7PT"
DMF_DATASET = "datasets/DMF"

EPSILON = 0.05
TEMPERATURE = 0.5


class DMFPILDataset(Dataset):

    def __init__(
        self, root_dir, save_dir, ids, extensions=(".jpg", ".jpeg", ".png", ".bmp")
    ):

        self.root_dir = root_dir
        self.extensions = extensions
        self.image_paths = [
            os.path.join(root, fname)
            for root, _, files in os.walk(root_dir)
            for fname in files
            if fname.lower().endswith(extensions) and (fname.split(".")[0] in ids)
        ]

        print(len(self.image_paths))

        if not self.image_paths:
            raise ValueError(
                f"No images found in {root_dir} with extensions {extensions}"
            )

        for idx, path in enumerate(self.image_paths):
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224), Image.BILINEAR)

            new_path = os.path.join(save_dir, os.path.basename(path))
            self.image_paths[idx] = new_path

            img.save(new_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return (img, self.image_paths[idx])


def get_image_dataloader_dmf(
    root_dir, save_dir, ids, batch_size=8, num_workers=4, shuffle=False
):

    assert Path(root_dir).is_dir()
    assert Path(save_dir).is_dir()

    dataset = DMFPILDataset(root_dir, save_dir, ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return loader


class HAM10000PILDataset(Dataset):

    def __init__(self, root_dir, save_dir, ids, extensions=(".jpg", ".jpeg", ".png", ".bmp")):

        self.root_dir = root_dir
        self.extensions = extensions
        self.image_paths = [
            os.path.join(root, fname)
            for root, _, files in os.walk(root_dir)
            for fname in files
            if fname.lower().endswith(extensions) and fname.startswith(ids)
        ]

        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir} with extensions {extensions}")
        
        for idx, path in enumerate(self.image_paths):
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224), Image.BILINEAR)
            
            new_path = os.path.join(save_dir, os.path.basename(path))
            self.image_paths[idx] = new_path

            img.save(new_path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()/255.0
        return (img, self.image_paths[idx])

def get_image_dataloader_ham(root_dir, save_dir, ids, batch_size=8, num_workers=4, shuffle=False):

    assert Path(root_dir).is_dir()
    assert Path(save_dir).is_dir()
    
    dataset = HAM10000PILDataset(root_dir, save_dir, ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return loader

class DERM7PTPILDataset(Dataset):

    def __init__(self, root_dir, save_dir, test_ids, meta, extensions=(".jpg", ".jpeg", ".png", ".bmp")):

        self.root_dir = root_dir
        self.extensions = extensions
        self.image_paths = [
            os.path.join(os.path.join(root_dir, "images"), meta.iloc[case_num]['derm']) 
            for case_num in test_ids['image_id']
        ]


        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir} with extensions {extensions}")
        
        for idx, path in enumerate(self.image_paths):
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224), Image.BILINEAR)
            
            new_path = os.path.join(save_dir, os.path.basename(path))
            self.image_paths[idx] = new_path

            img.save(new_path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()/255.0
        return (img, self.image_paths[idx])

def get_image_dataloader_d7p(root_dir, save_dir, ids, meta, batch_size=8, num_workers=4, shuffle=False):

    assert Path(root_dir).is_dir()
    assert Path(save_dir).is_dir()
    
    dataset = DERM7PTPILDataset(root_dir, save_dir, ids, meta)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return loader

def get_dataset_loader(dataset_name, batch_size=1):
    if dataset_name == "dmf":
        ids_path = os.path.join(TEST_IDS_PATH, "dmf_ids.csv")
        root_dir = DMF_DATASET
        save_dir = os.path.join(TEST_IMAGES_PATH, "dmf")
        os.makedirs(save_dir, exist_ok=True)
        ids_df = pd.read_csv(ids_path)
        ids = tuple(list(element.item() for element in ids_df.to_numpy()))
        return get_image_dataloader_dmf(root_dir, save_dir, ids=ids, batch_size=batch_size)
        
    elif dataset_name == "ham":
        ids_path = os.path.join(TEST_IDS_PATH, "ham_ids.csv")
        root_dir = HAM_DATASET
        save_dir = os.path.join(TEST_IMAGES_PATH, "ham")
        os.makedirs(save_dir, exist_ok=True)
        ids_df = pd.read_csv(ids_path)
        ids = tuple(list(element.item() for element in ids_df.to_numpy()))
        return get_image_dataloader_ham(root_dir, save_dir, ids=ids, batch_size=batch_size)
        
    elif dataset_name == "derm7pt":
        ids_path = os.path.join(TEST_IDS_PATH, "d7p_ids.csv")
        root_dir = DERM7PT_DATASET
        save_dir = os.path.join(TEST_IMAGES_PATH, "derm7pt")
        os.makedirs(save_dir, exist_ok=True)
        ids_df = pd.read_csv(ids_path)
        meta = pd.read_csv(os.path.join(DERM7PT_DATASET, "meta/meta.csv"))
        return get_image_dataloader_d7p(root_dir, save_dir, ids=ids_df, meta=meta, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# -----------------------
# Utilities: DCT basis (torch) & apply alphas in batch
# -----------------------
def make_low_freq_dct_basis_torch(shape, d, high_freq=False, device="cpu"):
    """
    Build DCT basis tensor (C*H*W, d) on given device (torch).
    Same basis as your numpy function but vectorized into a torch tensor.
    """
    C, H, W = shape
    basis = []
    max_f = int(math.sqrt(d)) + 3
    for u in range(max_f):
        for v in range(max_f):
            if len(basis) >= d:
                break
            grid = np.zeros((H, W), dtype=np.float32)
            # vectorized fill
            i_idx = np.arange(H).reshape(H, 1)
            j_idx = np.arange(W).reshape(1, W)
            grid = (
                np.cos(math.pi * u * (2 * i_idx + 1) / (2 * H))
                * np.cos(math.pi * v * (2 * j_idx + 1) / (2 * W))
            ).astype(np.float32)
            vec = np.tile(grid.flatten(), C)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            basis.append(vec)
        if len(basis) >= d:
            break
    B = np.stack(basis, axis=1).astype(np.float32)  # (CHW, d)
    if high_freq:
        B = B[:, ::-1]
    B_t = torch.from_numpy(B).to(device)  # (CHW, d)
    return B_t


def batch_apply_alphas(clean_img_tensor, B_t, alphas_np, eps=8 / 255.0):
    """
    Apply multiple alphas (numpy array shape (N, d)) to clean_img_tensor and return
    a torch tensor of images on the same device: (N, C, H, W).
    - clean_img_tensor: torch [C, H, W]
    - B_t: torch (CHW, d)
    - alphas_np: numpy (N, d)
    """
    device = B_t.device
    C, H, W = clean_img_tensor.shape
    N = alphas_np.shape[0]
    # convert alphas to tensor on device (d, N) matmul friendly
    alphas_t = torch.from_numpy(alphas_np.astype(np.float32)).to(device)  # (N, d)
    # compute delta_flat = alphas_t @ B_t^T  -> (N, CHW)
    # but more efficient: B_t (CHW,d) @ alphas_t.T (d,N) -> (CHW, N)
    delta_flat = B_t.matmul(alphas_t.t())  # (CHW, N)
    delta_flat = delta_flat.t()  # (N, CHW)
    imgs = delta_flat.view(N, C, H, W)
    imgs = torch.clamp(imgs, -eps, eps)
    clean = clean_img_tensor.unsqueeze(0).to(device)
    imgs = torch.clamp(clean + imgs, 0.0, 1.0)
    return imgs


# -----------------------
# Fast MOPSO (vectorized evaluation)
# -----------------------
class FastMOPSO:
    def __init__(
        self,
        clean_img,
        embedder,
        frd_factory,
        clean_batch,
        d=256,
        swarm_size=96,
        iters=400,
        alpha_scale=0.04,
        archive_size=50,
        frd_repeats=2,
        eps=8 / 255.0,
        tau=0.03,
        high_freq_basis=False,
        device="cuda",
        seed=0,
        verbose=True,
    ):
        self.device = torch.device(device)
        self.clean_img = clean_img.to(self.device).detach()
        self.embedder = embedder
        self.frd_fn = frd_factory(clean_batch.to(self.device))
        self.d = int(d)
        self.swarm_size = int(swarm_size)
        self.iters = int(iters)
        self.alpha_scale = float(alpha_scale)
        self.archive_size = int(archive_size)
        self.frd_repeats = int(frd_repeats)
        self.eps = float(eps)
        self.tau = float(tau)
        self.high_freq_basis = bool(high_freq_basis)
        self.verbose = verbose
        self.rng = np.random.RandomState(int(seed))
        # basis as torch (CHW, d)
        C, H, W = self.clean_img.shape
        self.B_t = make_low_freq_dct_basis_torch(
            (C, H, W), self.d, high_freq=self.high_freq_basis, device=self.device
        )
        # positions as numpy (swarm, d) for PSO math
        self.positions = (
            self.rng.randn(self.swarm_size, self.d).astype(np.float32)
            * self.alpha_scale
        )
        self.velocities = np.zeros_like(self.positions)
        self.pbest = self.positions.copy()
        self.pbest_scores = [
            None
        ] * self.swarm_size  # each score is np.array([frd, -adv])
        self.archive = []  # list of tuples (alpha_np, score_np)
        # bookkeeping
        self.query_count = 0
        self.progress_rows = []  # list of dicts for CSV
        self.best_adv_global = -np.inf
        self.no_improve_count = 0
        self.best_adv_iteration = 0
        if self.verbose:
            print(
                f"[FastMOPSO] init swarm={self.swarm_size} d={self.d} device={self.device}"
            )

    # try batch FRD evaluation; fall back to per-sample
    def _eval_frds_batch(self, imgs_batch):
        """
        imgs_batch: torch [N,C,H,W] on same device as frd_fn expects.
        Returns: numpy array (N,) FRD values.
        """
        # some user frd_fn accept batch: try it
        try:
            with torch.no_grad():
                vals = self.frd_fn(
                    imgs_batch
                )  # assume returns tensor like (N,) or list
            # convert to numpy
            if isinstance(vals, torch.Tensor):
                out = vals.detach().cpu().numpy().astype(np.float32)
            elif isinstance(vals, (list, tuple, np.ndarray)):
                out = np.array(vals, dtype=np.float32)
            else:
                # unexpected, fall back
                raise RuntimeError("frd_fn returned unsupported type")
            return out
        except Exception:
            # fallback per-sample (still keep tensors on device)
            outs = []
            N = imgs_batch.shape[0]
            with torch.no_grad():
                for i in range(N):
                    v = float(self.frd_fn(imgs_batch[i : i + 1]))
                    outs.append(v)
            return np.array(outs, dtype=np.float32)

    def _evaluate_swarm(self, positions_np):
        """
        Vectorized evaluation of the swarm:
        - Build images for all positions (batch)
        - Compute FRD (batch if possible)
        - Compute embedder adv distances (batch)
        Returns frd_vals (N,), adv_vals (N,), imgs_batch (torch)
        """
        N = positions_np.shape[0]
        # apply alphas in batch -> imgs (N,C,H,W) on device
        imgs = batch_apply_alphas(
            self.clean_img, self.B_t, positions_np, eps=self.eps
        )  # torch (N,C,H,W)
        # FRD evaluation (may be batched)
        frd_vals = None
        if self.frd_repeats <= 1:
            frd_vals = self._eval_frds_batch(imgs)
            self.query_count += N
        else:
            # repeat multiple times and average to reduce noise
            accum = np.zeros(N, dtype=np.float64)
            for _ in range(self.frd_repeats):
                out = self._eval_frds_batch(imgs)
                accum += out
                self.query_count += N
            frd_vals = (accum / float(self.frd_repeats)).astype(np.float32)
        # embedding distances (adv score) using embedder in batch
        with torch.no_grad():
            emb_clean = self.embedder.get_embeddings(
                self.clean_img.unsqueeze(0)
            )  # (1, D)
            emb_batch = self.embedder.get_embeddings(imgs)  # (N, D)
            # pairwise L2 distance between emb_clean and each emb_batch
            emb_clean_rep = emb_clean.repeat(N, 1)
            dists = (
                F.pairwise_distance(emb_clean_rep, emb_batch, p=2)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return frd_vals, dists, imgs

    @staticmethod
    def _score_from_vals(frd_arr, adv_arr):
        # score vector stored as [frd, -adv] for dominance ordering (minimize both)
        N = len(frd_arr)
        scores = np.stack([frd_arr, -adv_arr], axis=1).astype(np.float32)
        return scores

    @staticmethod
    def dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    def update_archive(self, cand_alphas, cand_scores):
        """
        cand_alphas: list of numpy arrays (k,d)
        cand_scores: numpy array (k,2)
        Maintains nondominated archive up to archive_size.
        """
        # merge
        all_alphas = [a for a, s in self.archive] + list(cand_alphas)
        all_scores = [s for a, s in self.archive] + list(cand_scores)
        keep = []
        for i, s in enumerate(all_scores):
            dominated = False
            for j, other in enumerate(all_scores):
                if j == i:
                    continue
                if np.all(other <= s) and np.any(other < s):
                    dominated = True
                    break
            if not dominated:
                keep.append((all_alphas[i], all_scores[i]))
        # sort by FRD asc, keep top archive_size
        keep_sorted = sorted(keep, key=lambda x: float(x[1][0]))
        self.archive = keep_sorted[: self.archive_size]

    def run(
        self, out_dir="mopso_out", early_stop_patience=20, early_stop_threshold=1e-4
    ):
        """
        Run MOPSO with early stopping.

        Args:
            early_stop_patience: number of iterations without improvement before stopping
            early_stop_threshold: minimum improvement in max_adv to count as progress
        """
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, CFG["progress_csv_name"])
        # CSV header
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iter",
                    "min_frd",
                    "max_adv",
                    "archive_size",
                    "queries",
                    "time_s",
                    "no_improve_count",
                ]
            )

        st = time.time()
        for it in range(1, self.iters + 1):
            # evaluate full swarm (vectorized)
            frd_vals, adv_vals, imgs_batch = self._evaluate_swarm(self.positions)
            # scores (k,2)
            scores = self._score_from_vals(frd_vals, adv_vals)
            # prepare candidate copies
            cand_alphas = [self.positions[i].copy() for i in range(self.swarm_size)]
            # update archive with this batch
            self.update_archive(cand_alphas, scores)

            # update personal bests (Pareto dominance)
            for i in range(self.swarm_size):
                cur = scores[i]
                if self.pbest_scores[i] is None:
                    self.pbest[i] = self.positions[i].copy()
                    self.pbest_scores[i] = cur.copy()
                else:
                    if self.dominates(cur, self.pbest_scores[i]):
                        self.pbest[i] = self.positions[i].copy()
                        self.pbest_scores[i] = cur.copy()
                    elif not self.dominates(self.pbest_scores[i], cur):
                        # none dominates -> heuristic: keep one with smaller FRD
                        if cur[0] < self.pbest_scores[i][0]:
                            self.pbest[i] = self.positions[i].copy()
                            self.pbest_scores[i] = cur.copy()

            # update velocities/positions
            for i in range(self.swarm_size):
                leader_alpha = self.select_leader()
                r1 = self.rng.rand(self.d).astype(np.float32)
                r2 = self.rng.rand(self.d).astype(np.float32)
                w, c1, c2 = 0.7, 1.5, 1.5
                self.velocities[i] = (
                    w * self.velocities[i]
                    + c1 * r1 * (self.pbest[i] - self.positions[i])
                    + c2 * r2 * (leader_alpha - self.positions[i])
                )
            self.positions += self.velocities
            # clip positions to keep numerical stability
            np.clip(self.positions, -0.5, 0.5, out=self.positions)

            # log statistics
            if len(self.archive) > 0:
                frd_vals_archive = [s[1][0] for s in self.archive]
                adv_vals_archive = [-s[1][1] for s in self.archive]
                min_frd = float(np.min(frd_vals_archive))
                max_adv = float(np.max(adv_vals_archive))
            else:
                min_frd = float(np.min(frd_vals))
                max_adv = float(np.max(adv_vals))

            # Early stopping: check for improvement
            if max_adv > self.best_adv_global + early_stop_threshold:
                self.best_adv_global = max_adv
                self.no_improve_count = 0
                self.best_adv_iteration = it
            else:
                self.no_improve_count += 1

            elapsed = time.time() - st
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        it,
                        min_frd,
                        max_adv,
                        len(self.archive),
                        int(self.query_count),
                        elapsed,
                        self.no_improve_count,
                    ]
                )
            if self.verbose and (it % max(1, self.iters // 20) == 0 or it == 1):
                print(
                    f"[iter {it}/{self.iters}] min_frd={min_frd:.6f} max_adv={max_adv:.4f} archive={len(self.archive)} queries={self.query_count} no_improve={self.no_improve_count}"
                )

            # Early stopping check
            if self.no_improve_count >= early_stop_patience:
                if self.verbose:
                    print(
                        f"[Early stopping] No improvement for {early_stop_patience} iterations. Stopping at iteration {it}."
                    )
                break

            # optional periodic save of top archive images
            if (it % CFG["save_every"] == 0 or it == self.iters) and len(
                self.archive
            ) > 0:
                self._save_archive_images(out_dir, it)

        # final save
        self._save_archive_images(out_dir, "final")
        print("[DONE] FastMOPSO finished. queries:", self.query_count)
        return self.archive

    def select_leader(self):
        if len(self.archive) == 0:
            return self.positions[self.rng.randint(self.swarm_size)].copy()
        idx = self.rng.randint(len(self.archive))
        return self.archive[idx][0].copy()

    def _save_archive_images(self, out_dir, iter_tag):
        # save top-k images from archive (on device -> CPU)
        topk = min(8, len(self.archive))
        for k in range(topk):
            alpha_np, score = self.archive[k]
            frd_val = float(score[0])
            adv_val = float(-score[1])
            # synthesize image for alpha (single)
            alphas = np.expand_dims(alpha_np, 0)
            with torch.no_grad():
                img_t = batch_apply_alphas(
                    self.clean_img, self.B_t, alphas, eps=self.eps
                )[0]
            arr = img_t.detach().cpu().numpy()
            arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(
                os.path.join(
                    out_dir,
                    f"iter{iter_tag}_top{k}_frd{frd_val:.4f}_adv{adv_val:.4f}.png",
                )
            )


# -----------------------
# Experiment runner + analysis
# -----------------------
def run_experiment():
    args = get_args()
    
    # Update CFG with args
    CFG["dataset_path"] = f"test_sets/test_images/{args.dataset}"
    if args.output_dir:
        CFG["output_dir"] = args.output_dir
    else:
        CFG["output_dir"] = f"results/{args.dataset}/pso"
        
    CFG["num_samples"] = args.num_samples
    CFG["swarm_size"] = args.swarm_size
    CFG["iters"] = args.iters
    CFG["device"] = args.device
    CFG["progress_csv_name"] = f"{args.dataset}_progress.csv"
    
    print(f"Running with config: {CFG}")

    # load dataset and embedder
    loader = get_dataset_loader(args.dataset)
    dataset = loader.dataset
    embedder = CLIPImageEmbedder(device=CFG["device"])
    frd_factory = get_distance_fn
    device_local = torch.device(CFG["device"])
    clean_batch = torch.stack(
        [dataset[i][0] for i in range(min(8, len(dataset)))], 0
    ).to(device_local)

    out_root = CFG["output_dir"]
    os.makedirs(out_root, exist_ok=True)

    # Load existing results if available
    summary_csv_path = os.path.join(out_root, "summary_all.csv")
    existing_results = {}
    if os.path.exists(summary_csv_path):
        try:
            df = pd.read_csv(summary_csv_path)
            existing_results = {int(row["idx"]): row for _, row in df.iterrows()}
            print(
                f"[Resume] Loaded {len(existing_results)} existing results from {summary_csv_path}"
            )
        except Exception as e:
            print(f"[Warning] Could not load existing results: {e}")

    overall_results = []

    num_samples = min(CFG["num_samples"], len(dataset))
    for idx in range(36, num_samples):
        # Skip if sample already completed
        if idx in existing_results:
            print(f"\n[Skip] Sample {idx+1}/{num_samples}: Already exists in results.")
            overall_results.append(existing_results[idx])
            continue

        sample_img, path = dataset[idx]
        sample_img = sample_img.to(device_local)
        out_dir = os.path.join(out_root, f"sample_{idx}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Sample {idx+1}/{num_samples}: {path} ===")
        mopso = FastMOPSO(
            clean_img=sample_img,
            embedder=embedder,
            frd_factory=frd_factory,
            clean_batch=clean_batch,
            d=CFG["d"],
            swarm_size=CFG["swarm_size"],
            iters=CFG["iters"],
            alpha_scale=CFG["alpha_scale"],
            archive_size=CFG["archive_size"],
            frd_repeats=CFG["frd_repeats"],
            eps=CFG["eps"],
            tau=CFG["tau"],
            high_freq_basis=CFG["high_freq_basis"],
            device=CFG["device"],
            seed=CFG["seed"],
            verbose=True,
        )

        start = time.time()
        archive = mopso.run(
            out_dir=out_dir, early_stop_patience=20, early_stop_threshold=1e-4
        )
        elapsed = time.time() - start

        # collect archive stats
        if len(archive) > 0:
            frds = [float(s[0]) for _, s in archive]
            advs = [-float(s[1]) for _, s in archive]
            stats = {
                "idx": idx,
                "path": path,
                "best_frd": float(np.min(frds)),
                "best_adv": float(np.max(advs)),
                "mean_frd": float(np.mean(frds)),
                "mean_adv": float(np.mean(advs)),
                "archive_size": len(archive),
                "queries": int(mopso.query_count),
                "time_s": float(elapsed),
            }
        else:
            stats = {
                "idx": idx,
                "path": path,
                "best_frd": float("nan"),
                "best_adv": float("nan"),
                "mean_frd": float("nan"),
                "mean_adv": float("nan"),
                "archive_size": 0,
                "queries": int(mopso.query_count),
                "time_s": float(elapsed),
            }
        overall_results.append(stats)
        # save per-sample summary json
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print("Sample stats:", stats)

    # Save global summary CSV
    import pandas as pd

    df = pd.DataFrame(overall_results)
    df.to_csv(os.path.join(out_root, "summary_all.csv"), index=False)
    print("\nAll done. Summary saved to:", os.path.join(out_root, "summary_all.csv"))

    # Global aggregated Pareto across samples: read per-iteration progress CSVs if exist
    all_points = []
    for idx in range(num_samples):
        pdir = os.path.join(out_root, f"sample_{idx}")
        prog = os.path.join(pdir, CFG["progress_csv_name"])
        if os.path.exists(prog):
            try:
                p_df = pd.read_csv(prog)
                for _, row in p_df.iterrows():
                    all_points.append((float(row["min_frd"]), float(row["max_adv"])))
            except Exception:
                continue
    if len(all_points) > 0:
        pts = np.array(all_points)
        # pareto front plotting (maximize adv, minimize frd)
        pts_sorted = pts[pts[:, 0].argsort()]
        pareto_x = []
        pareto_y = []
        best_y = -np.inf
        for x, y in pts_sorted:
            if y > best_y:
                pareto_x.append(x)
                pareto_y.append(y)
                best_y = y
        plt.figure(figsize=(8, 6))
        plt.scatter(pts[:, 0], pts[:, 1], alpha=0.3, label="All iter points")
        plt.plot(pareto_x, pareto_y, "-r", lw=2, label="Pareto front")
        plt.axvline(CFG["tau"], color="orange", linestyle="--", label="tau (FRD)")
        plt.xlabel("FRD (lower=more feasible)")
        plt.ylabel("Adversarial effect (higher better)")
        plt.legend()
        plt.grid()
        plt.title("Global Pareto: FRD vs Adv")
        plt.savefig(os.path.join(out_root, "global_pareto.png"))
        plt.close()
        print("Saved global_pareto.png")

    return overall_results


if __name__ == "__main__":
    all_stats = run_experiment()
    print("Done. Results:", all_stats)
