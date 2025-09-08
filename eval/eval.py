import argparse

import folium
import leafmap.foliumap as leafmap
import pandas as pd
from geopy.distance import geodesic
from prettytable import PrettyTable


# Fast local projection helpers (lat/lon -> meters using equirectangular approx)
def _compute_projection_params(df_groundtruth, df_predictions):
    lat0 = pd.concat([df_groundtruth["tree_lat"], df_predictions["tree_lat"]]).mean()
    lon0 = pd.concat([df_groundtruth["tree_lng"], df_predictions["tree_lng"]]).mean()
    # Meters per degree (approx near lat0)
    meters_per_deg_lat = 111_132.0
    from math import cos, radians

    meters_per_deg_lon = 111_320.0 * cos(radians(lat0))
    return lat0, lon0, meters_per_deg_lat, meters_per_deg_lon


def _project_inplace(df, lat0, lon0, mlat, mlon):
    # Adds x,y columns in meters relative to (lat0,lon0)
    df["x"] = (df["tree_lng"] - lon0) * mlon
    df["y"] = (df["tree_lat"] - lat0) * mlat


# Function to check if the ground truth and prediction coordinates are a match
def get_match(gt_coord, pred_coord, df_groundtruth, df_predictions):
    gt_pano_id = df_groundtruth[
        (df_groundtruth["tree_lat"] == gt_coord[0])
        & (df_groundtruth["tree_lng"] == gt_coord[1])
    ]["pano_id"].values[0]

    pred_pano_id = df_predictions[
        (df_predictions["tree_lat"] == pred_coord[0])
        & (df_predictions["tree_lng"] == pred_coord[1])
    ]["pano_id"].values[0]

    return gt_pano_id == pred_pano_id


# Function to remove duplicates within threshold distance (meters) using grid hashing
def remove_duplicates(df, threshold=1):
    if len(df) == 0:
        return df
    # Expect projected columns present
    if "x" not in df.columns or "y" not in df.columns:
        # Fallback to original O(n^2) if projection not available
        coords = df[["tree_lat", "tree_lng"]].to_numpy()
        to_remove = set()
        for i in range(len(coords)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(coords)):
                if j in to_remove:
                    continue
                distance = geodesic(coords[i], coords[j]).meters
                if distance <= threshold:
                    to_remove.add(j)
        return df.drop(index=list(to_remove)).reset_index(drop=True)

    cell_size = max(threshold, 1e-6)
    from math import floor

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    cell_to_indices = {}
    for idx, (xi, yi) in enumerate(zip(x, y)):
        cx = floor(xi / cell_size)
        cy = floor(yi / cell_size)
        cell_to_indices.setdefault((cx, cy), []).append(idx)

    to_remove = set()
    threshold_sq = float(threshold) * float(threshold)
    for (cx, cy), indices in cell_to_indices.items():
        # Compare within cell and 8 neighbors
        neighbor_cells = [
            (cx + dx, cy + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]
        # Greedy keep-first policy
        kept = []
        for i in indices:
            if i in to_remove:
                continue
            keep = True
            for k in kept:
                dx = x[i] - x[k]
                dy = y[i] - y[k]
                if dx * dx + dy * dy <= threshold_sq:
                    keep = False
                    break
            if keep:
                kept.append(i)
            else:
                to_remove.add(i)

    return df.drop(index=list(to_remove)).reset_index(drop=True)


# Function to plot the ground truth and predictions
def plot_map(df_groundtruth, df_predictions, tp_matches, zoom_start=15):
    map_center = [
        df_groundtruth["tree_lat"].mean(),
        df_groundtruth["tree_lng"].mean(),
    ]

    # Create a leafmap object (which is Folium-based) and add hybrid tiles
    fmap = leafmap.Map(center=map_center, zoom=zoom_start, max_zoom=25)

        # Highlight true positives with a circle around both points
    for gt_coord, pred_coord in tp_matches:
        radius = geodesic(gt_coord, pred_coord).meters / 2 + 2

        midpoint = [
            (gt_coord[0] + pred_coord[0]) / 2,
            (gt_coord[1] + pred_coord[1]) / 2,
        ]

        color = (
            "blue"
            if get_match(gt_coord, pred_coord, df_groundtruth, df_predictions)
            else "purple"
        )

        folium.Circle(
            location=midpoint,
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.2,
        ).add_to(fmap)

    for _, row in df_predictions.iterrows():
        image_url = f"http://localhost:8000/{row['image_path'].split('/')[-1]}"  # Adjust if hosted elsewhere
        popup_html = f"""
        <div>
            <p><b>Pano ID:</b> {row["pano_id"]}</p>
            <img src="{image_url}" width="300">
        </div>
        """

        folium.CircleMarker(
            location=[row["tree_lat"], row["tree_lng"]],
            popup=folium.Popup(popup_html, max_width=350),
            radius=0.2,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.7,
        ).add_to(fmap)

    # Add ground truth points
    for _, row in df_groundtruth.iterrows():
        folium.CircleMarker(
            location=[row["tree_lat"], row["tree_lng"]],
            popup=row["pano_id"],
            radius=0.2,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.7,
        ).add_to(fmap)



    esri_layer = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri WorldImagery',
        overlay=True,
        control=True,
        max_zoom=25
    )
    esri_layer.add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.save("map_with_matches.html")


def evaluate(df_groundtruth, df_predictions, threshold, plot):
    # Expect projected columns present
    gt_latlng = df_groundtruth[["tree_lat", "tree_lng"]].to_numpy()
    pr_latlng = df_predictions[["tree_lat", "tree_lng"]].to_numpy()
    gt_xy = df_groundtruth[["x", "y"]].to_numpy() if {"x", "y"}.issubset(df_groundtruth.columns) else None
    pr_xy = df_predictions[["x", "y"]].to_numpy() if {"x", "y"}.issubset(df_predictions.columns) else None

    if gt_xy is None or pr_xy is None:
        # Fallback to original method if projection missing
        groundtruth_coords = gt_latlng
        prediction_coords = pr_latlng
        tp_count = 0
        false_negatives = set(range(len(groundtruth_coords)))
        matched_predictions = set()
        tp_matches = []
        for i, gt_coord in enumerate(groundtruth_coords):
            min_distance = float("inf")
            best_match = None
            for j, pred_coord in enumerate(prediction_coords):
                if j in matched_predictions:
                    continue
                distance = geodesic(gt_coord, pred_coord).meters
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    best_match = j
            if best_match is not None:
                tp_count += 1
                matched_predictions.add(best_match)
                false_negatives.discard(i)
                tp_matches.append((gt_coord, prediction_coords[best_match]))
        false_positives = [j for j in range(len(prediction_coords)) if j not in matched_predictions]
        fn_count = len(false_negatives)
        fp_count = len(false_positives)
        precision = round(tp_count / (tp_count + fp_count), 2) if (tp_count + fp_count) > 0 else 0.0
        recall = round(tp_count / (tp_count + fn_count), 2) if (tp_count + fn_count) > 0 else 0.0
        f1_score = round(2 * tp_count / (2 * tp_count + fp_count + fn_count), 2) if (2 * tp_count + fp_count + fn_count) > 0 else 0.0
        match_count = 0
        for gt_coord, pred_coord in tp_matches:
            if get_match(gt_coord, pred_coord, df_groundtruth, df_predictions):
                match_count += 1
        matches_percentage = round(match_count / len(tp_matches) * 100, 2) if len(tp_matches) > 0 else 0.0
        if plot:
            plot_map(df_groundtruth, df_predictions, tp_matches)
        return precision, recall, f1_score, matches_percentage

    # Grid-based nearest unmatched match within threshold
    from math import floor

    cell_size = max(threshold, 1e-6)
    threshold_sq = float(threshold) * float(threshold)

    # Build grid for predictions
    cell_to_pred = {}
    for j, (xj, yj) in enumerate(pr_xy):
        cx = floor(xj / cell_size)
        cy = floor(yj / cell_size)
        cell_to_pred.setdefault((cx, cy), []).append(j)

    matched_predictions = set()
    tp_matches = []

    for i, (xg, yg) in enumerate(gt_xy):
        cgx = floor(xg / cell_size)
        cgy = floor(yg / cell_size)
        best_j = None
        best_d2 = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cell_to_pred.get((cgx + dx, cgy + dy), []):
                    if j in matched_predictions:
                        continue
                    xj, yj = pr_xy[j]
                    dxm = xg - xj
                    dym = yg - yj
                    d2 = dxm * dxm + dym * dym
                    if d2 <= threshold_sq and d2 < best_d2:
                        best_d2 = d2
                        best_j = j
        if best_j is not None:
            matched_predictions.add(best_j)
            tp_matches.append((gt_latlng[i], pr_latlng[best_j]))

    tp_count = len(tp_matches)
    fp_count = len(pr_latlng) - len(matched_predictions)
    fn_count = len(gt_latlng) - tp_count

    precision = round(tp_count / (tp_count + fp_count), 2) if (tp_count + fp_count) > 0 else 0.0
    recall = round(tp_count / (tp_count + fn_count), 2) if (tp_count + fn_count) > 0 else 0.0
    f1_score = round(2 * tp_count / (2 * tp_count + fp_count + fn_count), 2) if (2 * tp_count + fp_count + fn_count) > 0 else 0.0

    # Compute matches % without DataFrame scans
    gt_map = { (row[0], row[1]): pano for row, pano in zip(df_groundtruth[["tree_lat", "tree_lng"]].to_numpy(), df_groundtruth["pano_id"].to_numpy()) }
    pr_map = { (row[0], row[1]): pano for row, pano in zip(df_predictions[["tree_lat", "tree_lng"]].to_numpy(), df_predictions["pano_id"].to_numpy()) }
    match_count = 0
    for gt_coord, pred_coord in tp_matches:
        pano_gt = gt_map.get((gt_coord[0], gt_coord[1]))
        pano_pr = pr_map.get((pred_coord[0], pred_coord[1]))
        if pano_gt is not None and pano_pr is not None and pano_gt == pano_pr:
            match_count += 1
    matches_percentage = round(match_count / len(tp_matches) * 100, 2) if len(tp_matches) > 0 else 0.0

    if plot:
        plot_map(df_groundtruth, df_predictions, tp_matches)

    return precision, recall, f1_score, matches_percentage


def get_results(df_groundtruth, df_predictions, args):
    print(f"\nGround Truth: {len(df_groundtruth)}")
    print(f"Predictions: {len(df_predictions)}")

    max_precision = [0, []]
    max_recall = [0, []]
    max_f1_score = [0, []]

    # Define Evaluation Thresholds
    conf_thresholds = [0.2, 0.5, 0.7]
    distance_thresholds = [3, 5]
    duplicate_thresholds = [2, 5]
    table = PrettyTable(
        [
            "Metric",
            "Predictions",
            "Confidence",
            "Distance (m)",
            "Duplicate (m)",
            "Precision",
            "Recall",
            "F1 Score",
            "Matches %",
        ]
    )

    # Precompute local projection to meters and add x,y columns
    lat0, lon0, mlat, mlon = _compute_projection_params(df_groundtruth, df_predictions)
    _project_inplace(df_groundtruth, lat0, lon0, mlat, mlon)
    _project_inplace(df_predictions, lat0, lon0, mlat, mlon)

    # Compute Metrics for Each Threshold
    for conf_threshold in conf_thresholds:
        for dist_threshold in distance_thresholds:
            for duplicate_threshold in duplicate_thresholds:
                df_prediction = df_predictions[df_predictions["conf"] >= conf_threshold]
                df_prediction.reset_index(drop=True, inplace=True)
                df_prediction = remove_duplicates(df_prediction, duplicate_threshold)
                precision, recall, f1_score, matches = evaluate(
                    df_groundtruth, df_prediction, dist_threshold, args.plot
                )
                if f1_score > max_f1_score[0]:
                    max_f1_score[0] = f1_score
                    max_f1_score[1] = [
                        "F1 Score",
                        len(df_prediction),
                        conf_threshold,
                        dist_threshold,
                        duplicate_threshold,
                        precision,
                        recall,
                        f1_score,
                        matches,
                    ]
                if precision > max_precision[0]:
                    max_precision[0] = precision
                    max_precision[1] = [
                        "Precision",
                        len(df_prediction),
                        conf_threshold,
                        dist_threshold,
                        duplicate_threshold,
                        precision,
                        recall,
                        f1_score,
                        matches,
                    ]
                if recall > max_recall[0]:
                    max_recall[0] = recall
                    max_recall[1] = [
                        "Recall",
                        len(df_prediction),
                        conf_threshold,
                        dist_threshold,
                        duplicate_threshold,
                        precision,
                        recall,
                        f1_score,
                        matches,
                    ]

        table.add_row(max_precision[1])
        table.add_row(max_recall[1])
        table.add_row(max_f1_score[1])

        # Print Summary Table
        print(table)
        table.clear_rows()
        max_precision = [0, []]
        max_recall = [0, []]
        max_f1_score = [0, []]


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against ground truth."
    )
    parser.add_argument(
        "predictions_csv_path",
        type=str,
        help="Path to the CSV file containing predictions.",
    )
    parser.add_argument(
        "--plot",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Create plot (True/False). Default: False",
    )
    return parser


parser = create_parser()
args = parser.parse_args()
df_groundtruth = pd.read_csv("./28_29_groundtruth.csv")
df_predictions = pd.read_csv(args.predictions_csv_path)

get_results(df_groundtruth, df_predictions, args)
