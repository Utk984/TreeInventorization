import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# File containing panorama data
CSV_FILE = "./data/input/cdg_st_v3_28_29_panoramas.csv"


def load_panoramas_from_csv(filename):
    """Load panorama data from a CSV file."""
    try:
        panoramas = pd.read_csv(filename)
        return panoramas
    except FileNotFoundError:
        st.error(f"File not found: {filename}")
        return pd.DataFrame()


def create_map(panoramas):
    """Create a folium map with panorama markers."""
    if panoramas.empty:
        st.warning("No panorama data available to display.")
        return folium.Map(location=[30.717, 76.795], zoom_start=13)

    # Center the map around the average lat/lon
    avg_lat = panoramas["lat"].mean()
    avg_lon = panoramas["lon"].mean()
    panorama_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)

    for _, row in panoramas.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=2,  # Small radius for dots
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.6,
            ).add_to(panorama_map)

    return panorama_map


def main():
    st.title("Chandigarh Panorama Viewer")
    st.write("Visualize all panoramas loaded from the CSV file.")

    # Load panorama data
    panoramas = load_panoramas_from_csv(CSV_FILE)

    if not panoramas.empty:
        st.success(f"Loaded {len(panoramas)} panoramas from the CSV file.")
        st.write("Below is an interactive map of the panoramas.")

        # Create and display map
        panorama_map = create_map(panoramas)
        st_folium(panorama_map, width=800, height=600)
    else:
        st.warning("No data available to display on the map.")


if __name__ == "__main__":
    main()
