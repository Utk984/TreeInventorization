import argparse

import folium
import leafmap.foliumap as leafmap
import pandas as pd
from geopy.distance import geodesic
from prettytable import PrettyTable


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


# Function to remove duplicates within threshold distance
def remove_duplicates(df, threshold=1):
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


# Function to plot the ground truth and predictions
def plot_map(df_groundtruth, df_predictions, tp_matches, zoom_start=15):
    map_center = [
        df_groundtruth["tree_lat"].mean(),
        df_groundtruth["tree_lng"].mean(),
    ]

    # Create a leafmap object (which is Folium-based) and add hybrid tiles
    fmap = leafmap.Map(center=map_center, zoom=zoom_start, max_zoom=25)
    fmap.add_basemap("HYBRID")
    fmap.add_basemap("Esri.WorldImagery")

    for _, row in df_predictions.iterrows():
        image_url = f"http://localhost:8000/{row['image_path'].split('/')[-1]}"  # Adjust if hosted elsewhere
        popup_html = f"""
        <div>
            <p><b>Pano ID:</b> {row["pano_id"]}</p>
            <img src="{image_url}" width="300">
        </div>
        """

        folium.Marker(
            location=[row["tree_lat"], row["tree_lng"]],
            popup=folium.Popup(popup_html, max_width=350),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(fmap)

    # Add ground truth points
    for _, row in df_groundtruth.iterrows():
        folium.Marker(
            location=[row["tree_lat"], row["tree_lng"]],
            popup=row["pano_id"],
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(fmap)

    # for _, row in df_predictions.iterrows():
    #     folium.Marker(
    #         location=[row["tree_lat"], row["tree_lng"]],
    #         popup=row["pano_id"],
    #         icon=folium.Icon(color="red", icon="info-sign"),
    #     ).add_to(fmap)

    # coordinates = (
    #     row["coordinates"]
    #     .replace("[", "")
    #     .replace("]", "")
    #     .replace("(", "")
    #     .replace(")", "")
    #     .replace("None", "")
    #     .split(", ")
    # )
    #
    # for i in range(0, len(coordinates), 2):
    #     lat = coordinates[i].strip()
    #     lon = coordinates[i + 1].strip() if i + 1 < len(coordinates) else None
    #     if lat and lon:
    #         try:
    #             lat = float(lat)
    #             lon = float(lon)
    #         except ValueError:
    #             continue
    #         folium.CircleMarker(
    #             location=[float(lat), float(lon)],
    #             radius=2,
    #             color="red",
    #             fill=True,
    #             fill_color="red",
    #             opacity=0.5,
    #         ).add_to(fmap)

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
            popup="True Positive Pair",
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap.save("map_with_matches.html")


def evaluate(df_groundtruth, df_predictions, threshold, plot):
    groundtruth_coords = df_groundtruth[["tree_lat", "tree_lng"]].values
    prediction_coords = df_predictions[["tree_lat", "tree_lng"]].values

    tp_count = 0
    false_positives = []
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

    false_positives = [
        j for j in range(len(prediction_coords)) if j not in matched_predictions
    ]

    fn_count = len(false_negatives)
    fp_count = len(false_positives)
    precision = round(tp_count / (tp_count + fp_count), 2)
    recall = round(tp_count / (tp_count + fn_count), 2)
    f1_score = round(2 * tp_count / (2 * tp_count + fp_count + fn_count), 2)

    match_count = 0
    for gt_coord, pred_coord in tp_matches:
        if get_match(gt_coord, pred_coord, df_groundtruth, df_predictions):
            match_count += 1

    matches_percentage = round(match_count / len(tp_matches) * 100, 2)

    if plot:
        plot_map(df_groundtruth, df_predictions, tp_matches)

    return precision, recall, f1_score, matches_percentage


def get_results(df_groundtruth, df_predictions, args):
    print(f"\nGround Truth: {len(df_groundtruth)}")

    max_precision = [0, []]
    max_recall = [0, []]
    max_f1_score = [0, []]

    # Define Evaluation Thresholds
    conf_thresholds = [0.1, 0.3, 0.5]
    distance_thresholds = [3, 5, 7]
    duplicate_thresholds = [0, 1, 2]
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
        "--duplicate",
        type=float,
        default=0,
        help="Remove duplicates within threshold (meters). Default: None",
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
