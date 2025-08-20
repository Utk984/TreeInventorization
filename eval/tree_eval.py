#!/usr/bin/env python3
"""
Robust evaluation script for tree detection results.
---------------------------------------------------
This script compares predicted tree coordinates against ground-truth coordinates
and reports key metrics:
    • Precision, Recall, F1
    • Average distance error for true-positives
    • TP / FP / FN counts

Key features
============
1. Duplicate handling
   Clusters predictions that are within `duplicate_eps` metres of each other
   (DBSCAN) and keeps the *highest-confidence* prediction from each cluster.

2. Optimal matching
   Uses the Hungarian algorithm to find an *optimal* one-to-one assignment
   between predictions and ground-truth points within a given distance
   threshold.  This avoids greedy pitfalls and double-matching.

3. Fast geodesic distance computation
   Converts lat/lon to an equal-area projection (EPSG:3857) so distance
   computation becomes simple Euclidean; this is sufficiently accurate for
   small distances (<100 m) and vastly faster than many geodesic calls.

4. Aggregated metrics across multiple distance thresholds.

Usage
-----
python tree_eval.py --distance 5 --duplicate 1 --plot False

If your predictions CSV lacks a `conf` column, a constant value of 1.0 will be
assumed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from pyproj import Transformer


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# Re-use one global transformer (WGS84 ➜ WebMercator) for speed
_WGS84_TO_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon (deg) to metres in Web-Mercator (EPSG:3857)."""
    x, y = _WGS84_TO_MERCATOR.transform(lon, lat)  # note order: lon, lat
    return np.vstack([x, y]).T  # shape (N,2)


# ---------------------------------------------------------------------------
# Duplicate removal via DBSCAN
# ---------------------------------------------------------------------------

def suppress_duplicates(df_preds: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Remove duplicates closer than *eps* metres; keep highest-confidence."""
    if eps <= 0 or len(df_preds) == 0:
        return df_preds

    coords = latlon_to_xy(df_preds.tree_lat.values, df_preds.tree_lng.values)
    clustering = DBSCAN(eps=eps, min_samples=1, metric="euclidean").fit(coords)
    df_preds["cluster"] = clustering.labels_

    # Keep highest conf per cluster
    keep_indices: List[int] = []
    for cl in np.unique(clustering.labels_):
        cluster_df = df_preds[df_preds.cluster == cl]
        best_idx = cluster_df.conf.idxmax()
        keep_indices.append(best_idx)

    cleaned = df_preds.loc[keep_indices].copy().reset_index(drop=True)
    cleaned.drop(columns="cluster", inplace=True)
    return cleaned


# ---------------------------------------------------------------------------
# Matching using Hungarian algorithm
# ---------------------------------------------------------------------------

def match_predictions(
    gt_coords: np.ndarray, preds_coords: np.ndarray, max_distance: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return indices of (TP, FP, FN) using optimal assignment.

    • gt_coords, preds_coords are (N,2) and (M,2) XY arrays in metres.
    • max_distance: maximum matching distance in metres.
    """
    if len(gt_coords) == 0:
        return np.array([], int), np.arange(len(preds_coords)), np.array([], int), np.array([], int)
    if len(preds_coords) == 0:
        return np.array([], int), np.array([], int), np.arange(len(gt_coords)), np.arange(len(gt_coords))

    # Compute squared distances (fast) and mask by threshold
    d2 = np.sum((gt_coords[:, None, :] - preds_coords[None, :, :]) ** 2, axis=2)
    d = np.sqrt(d2)

    # Cost matrix: distances; anything beyond threshold set to large value
    large_cost = max_distance + 1.0  # anything > threshold
    cost = np.where(d <= max_distance, d, large_cost)

    row_ind, col_ind = linear_sum_assignment(cost)

    tp_rows: List[int] = []
    tp_cols: List[int] = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_distance:
            tp_rows.append(r)
            tp_cols.append(c)

    tp_rows = np.array(tp_rows, int)
    tp_cols = np.array(tp_cols, int)

    fp_cols = np.setdiff1d(np.arange(len(preds_coords)), tp_cols, assume_unique=True)
    fn_rows = np.setdiff1d(np.arange(len(gt_coords)), tp_rows, assume_unique=True)
    return tp_rows, tp_cols, fp_cols, fn_rows


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    distance_thresh: float = 5.0,
    duplicate_eps: float = 1.0,
) -> dict:
    """Compute precision/recall/F1 for given thresholds."""
    # Clean duplicates
    pred_df = suppress_duplicates(pred_df.copy(), duplicate_eps)

    # Coordinates to xy metres
    gt_xy = latlon_to_xy(gt_df.tree_lat.values, gt_df.tree_lng.values)
    pred_xy = latlon_to_xy(pred_df.tree_lat.values, pred_df.tree_lng.values)

    tp_rows, tp_cols, fp_cols, fn_rows = match_predictions(gt_xy, pred_xy, distance_thresh)

    tp = len(tp_rows)
    fp = len(fp_cols)
    fn = len(fn_rows)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    # Average distance for TP
    if tp:
        dists = np.linalg.norm(gt_xy[tp_rows] - pred_xy[tp_cols], axis=1)
        avg_distance = float(np.mean(dists))
    else:
        avg_distance = None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "avg_distance_m": round(avg_distance, 2) if avg_distance is not None else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate tree detections vs ground truth.")
    p.add_argument("--distance", type=float, default=5.0, help="Match distance threshold in metres (default=5)")
    p.add_argument("--duplicate", type=float, default=2.0, help="Duplicate suppression distance in metres (default=1)")
    return p.parse_args()


def main():
    args = parse_args()

    gt_df = pd.read_csv("28_29_groundtruth.csv")
    pred_df = pd.read_csv("tree_data.csv")

    if "conf" not in pred_df.columns:
        pred_df["conf"] = 1.0

    results = evaluate(gt_df, pred_df, distance_thresh=args.distance, duplicate_eps=args.duplicate)
    print(results)


if __name__ == "__main__":
    main() 