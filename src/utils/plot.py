import pandas as pd
import folium
from folium import plugins
import numpy as np
from geopy.distance import geodesic
import os
import sys
from typing import List, Tuple, Dict
import argparse
from flask import Flask, render_template_string


class TreeStreetViewPlotter:
    """Main class for plotting trees and street views with interactive features."""
    
    def __init__(self, tree_csv_path: str, streetview_csv_path: str, 
                 groundtruth_csv_path: str,
                 data_dir: str = "data", server_url: str = "http://localhost:8000"):
        """
        Initialize the plotter with data paths and server configuration.
        
        Args:
            tree_csv_path: Path to tree data CSV file
            streetview_csv_path: Path to street view CSV file
            groundtruth_csv_path: Path to ground truth CSV file
            data_dir: Directory containing the data folder
            server_url: URL for serving static files
        """
        self.tree_csv_path = tree_csv_path
        self.streetview_csv_path = streetview_csv_path
        self.groundtruth_csv_path = groundtruth_csv_path
        self.data_dir = data_dir
        self.server_url = server_url
        self.tree_data = None
        self.streetview_data = None
        self.cleaned_tree_data = None
        self.groundtruth_data = None

    def load_data(self):
        """Load tree and street view data from CSV files."""
        print("Loading data...")
        self.tree_data = pd.read_csv(self.tree_csv_path)
        # keep the rows where distance_pano < 10
        self.tree_data = self.tree_data[self.tree_data['distance_pano'] < 12]
        self.streetview_data = pd.read_csv(self.streetview_csv_path)
        print(f"Loaded {len(self.tree_data)} tree records")
        print(f"Loaded {len(self.streetview_data)} street view records")
        self.groundtruth_data = pd.read_csv(self.groundtruth_csv_path)
        print(f"Loaded {len(self.groundtruth_data)} ground truth records")

    def remove_duplicate_trees(self, distance_threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove duplicate trees based on distance threshold.
        
        Args:
            distance_threshold: Distance in meters to consider trees as duplicates
            
        Returns:
            DataFrame with duplicate trees removed
        """
        print(f"Removing duplicate trees within {distance_threshold}m...")
        
        # Get unique tree coordinates
        tree_coords = self.tree_data[['tree_lat', 'tree_lng']].drop_duplicates()
        tree_coords = tree_coords.dropna()
        
        if len(tree_coords) == 0:
            print("No valid tree coordinates found")
            return pd.DataFrame()
        
        # Convert to list of tuples for easier processing
        coords_list = [(row['tree_lat'], row['tree_lng']) for _, row in tree_coords.iterrows()]
        
        # Find unique coordinates
        unique_coords = []
        removed_count = 0
        
        for i, coord1 in enumerate(coords_list):
            is_duplicate = False
            for coord2 in unique_coords:
                # Calculate distance in meters
                distance = geodesic(coord1, coord2).meters
                if distance <= distance_threshold:
                    is_duplicate = True
                    removed_count += 1
                    break
            
            if not is_duplicate:
                unique_coords.append(coord1)
        
        print(f"Removed {removed_count} duplicate trees, kept {len(unique_coords)} unique trees")
        
        # Filter original data to keep only unique trees
        unique_lat_lng = set(unique_coords)
        mask = self.tree_data.apply(
            lambda row: (row['tree_lat'], row['tree_lng']) in unique_lat_lng, 
            axis=1
        )
        
        self.cleaned_tree_data = self.tree_data[mask].copy()
        return self.cleaned_tree_data
    
    def create_image_popup(self, image_path: str, title: str = "") -> str:
        """
        Create HTML popup content with image.
        
        Args:
            image_path: Path to the image file
            title: Title for the popup
            
        Returns:
            HTML string for popup
        """
        # Extract filename using split('/')[-1]
        filename = image_path.split('/')[-1]
        
        # Determine the correct path based on filename
        if 'view' in filename:
            # Tree image - use views directory
            image_url = f"{self.server_url}/views/{filename}"
        else:
            # Street view image - use full directory
            image_url = f"{self.server_url}/full/{filename}"
        html = f"""
        <div style="text-align: center;">
            <h4>{title}</h4>
            <img src="{image_url}" style="max-width: 300px; max-height: 200px; border-radius: 5px;">
        </div>
        """
        return html
    
    def create_map(self) -> folium.Map:
        """
        Create the interactive map with trees, street views, and connections.
        
        Returns:
            folium.Map object
        """
        print("Creating interactive map...")
        
        # Calculate center point
        all_lats = list(self.cleaned_tree_data['tree_lat']) + list(self.streetview_data['lat'])
        all_lngs = list(self.cleaned_tree_data['tree_lng']) + list(self.streetview_data['lng'])
        
        center_lat = np.mean(all_lats)
        center_lng = np.mean(all_lngs)
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=15,
            tiles='OpenStreetMap',
            width='100%',
            height='100%',
            max_zoom=25,
        )
        
        # Add tree markers
        print("Adding tree markers...")
        for _, row in self.cleaned_tree_data.iterrows():
            if pd.notna(row['tree_lat']) and pd.notna(row['tree_lng']):
                # Create popup with tree image
                popup_html = self.create_image_popup(
                    row['image_path'], 
                    f"Tree at ({row['tree_lat']:.6f}, {row['tree_lng']:.6f})"
                )
                
                folium.CircleMarker(
                    [row['tree_lat'], row['tree_lng']],
                    popup=folium.Popup(popup_html, max_width=400),
                    radius=0.2,
                    color='green',
                    fill=True,
                    fillColor='green',
                    fillOpacity=0.8
                ).add_to(m)
        
        # Add street view markers
        print("Adding street view markers...")
        for _, row in self.streetview_data.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['lng']):
                # Create popup with street view image
                streetview_image_path = os.path.join(self.data_dir, 'full', f"{row['pano_id']}.jpg")
                popup_html = self.create_image_popup(
                    streetview_image_path,
                    f"Street View at ({row['lat']:.6f}, {row['lng']:.6f})"
                )
                
                folium.CircleMarker(
                    [row['lat'], row['lng']],
                    popup=folium.Popup(popup_html, max_width=400),
                    radius=0.2,
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.8
                ).add_to(m)

        # Add ground truth markers
        print("Adding ground truth markers...")
        for _, row in self.groundtruth_data.iterrows():
            if pd.notna(row['tree_lat']) and pd.notna(row['tree_lng']):
                folium.CircleMarker(
                    [row['tree_lat'], row['tree_lng']],
                    popup=folium.Popup(f"Ground Truth at ({row['tree_lat']:.6f}, {row['tree_lng']:.6f})"),
                    radius=0.2,
                    color='purple',
                    fill=True,
                    fillColor='purple',
                    fillOpacity=0.8
                ).add_to(m)
        
        # Add connection lines between trees and street views
        print("Adding connection lines...")
        self._add_connection_lines(m)
        
        # Add legend
        self._add_legend(m)
        
        return m
    
    def _add_connection_lines(self, map_obj: folium.Map):
        """Add lines connecting trees to their corresponding street views."""
        # Group trees by pano_id
        tree_groups = self.cleaned_tree_data.groupby('pano_id')
        
        for pano_id, tree_group in tree_groups:
            # Find corresponding street view
            streetview = self.streetview_data[self.streetview_data['pano_id'] == pano_id]
            
            if len(streetview) > 0:
                sv_lat = streetview.iloc[0]['lat']
                sv_lng = streetview.iloc[0]['lng']
                
                # Draw lines from street view to each tree
                for _, tree in tree_group.iterrows():
                    if pd.notna(tree['tree_lat']) and pd.notna(tree['tree_lng']):
                        folium.PolyLine(
                            locations=[[sv_lat, sv_lng], [tree['tree_lat'], tree['tree_lng']]],
                            color='red',
                            weight=1,
                            opacity=0.6,
                            popup=f"Connection: {pano_id}"
                        ).add_to(map_obj)
    
    def _add_legend(self, map_obj: folium.Map):
        """Add legend to the map."""
        legend_html = '''
        <div style="position: absolute; 
                    bottom: 10px; right: 10px; width: 140px; height: 80px; 
                    background-color: white; border:2px solid grey; z-index:1000; 
                    font-size:12px; padding: 8px; border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        <p style="margin: 0 0 5px 0; font-weight: bold;">Legend</p>
        <p style="margin: 2px 0;"><span style="color:green">●</span> Trees</p>
        <p style="margin: 2px 0;"><span style="color:blue">●</span> Street Views</p>
        <p style="margin: 2px 0;"><span style="color:red">━</span> Connections</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def get_map_html(self) -> str:
        """Get the map as HTML string for streaming."""
        return self.map_obj._repr_html_()
    
    def run(self, distance_threshold: float = 3.0):
        """
        Run the complete plotting pipeline.
        
        Args:
            distance_threshold: Distance threshold for duplicate removal in meters
        """
        try:
            # Load data
            self.load_data()
            
            # Remove duplicates
            self.cleaned_tree_data = self.remove_duplicate_trees(distance_threshold)
            
            if len(self.cleaned_tree_data) == 0:
                print("No tree data to plot after duplicate removal")
                return None
            
            # Create map
            self.map_obj = self.create_map()
            
            print(f"\nMap created successfully!")
            print(f"Map will be served at http://localhost:5000")
            print(f"Make sure the data server is running on {self.server_url}")
            
            return self.map_obj
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

def create_flask_app(plotter):
    """Create Flask app to serve the map."""
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        map_html = plotter.get_map_html()
        return render_template_string(
            '<!DOCTYPE html>'
            '<html>'
            '<head>'
            '    <title>Tree and Street View Map</title>'
            '    <meta charset="utf-8">'
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">'
            '    <style>'
            '        * {'
            '            margin: 0;'
            '            padding: 0;'
            '            box-sizing: border-box;'
            '        }'
            '        body, html {'
            '            height: 100%;'
            '            width: 100%;'
            '            overflow: hidden;'
            '            position: fixed;'
            '        }'
            '        .folium-map {'
            '            height: 100vh !important;'
            '            width: 100vw !important;'
            '            position: fixed !important;'
            '            top: 0;'
            '            left: 0;'
            '        }'
            '        .leaflet-container {'
            '            height: 100vh !important;'
            '            width: 100vw !important;'
            '        }'
            '        .leaflet-control-container {'
            '            z-index: 1001;'
            '        }'
            '    </style>'
            '</head>'
            '<body>'
            '    {{ map_html | safe }}'
            '</body>'
            '</html>',
            map_html=map_html
        )
    
    return app

def main():
    """Main function to run the plotter from command line."""
    parser = argparse.ArgumentParser(description='Plot trees and street views on interactive map')
    parser.add_argument('--tree-csv', default='outputs/chandigarh_trees.csv', help='Path to tree data CSV file')
    parser.add_argument('--streetview-csv', default='streetviews/chandigarh_streets.csv', help='Path to street view CSV file')
    parser.add_argument('--groundtruth-csv', default='eval/chandigarh_groundtruth.csv', help='Path to ground truth CSV file')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--server-url', default='http://localhost:8000', help='Server URL for images')
    parser.add_argument('--distance-threshold', type=float, default=3.0, 
                       help='Distance threshold for duplicate removal in meters')
    parser.add_argument('--port', type=int, default=5001, help='Port to serve the map')
    
    args = parser.parse_args()
    
    # Create and run plotter
    plotter = TreeStreetViewPlotter(
        tree_csv_path=args.tree_csv,
        streetview_csv_path=args.streetview_csv,
        groundtruth_csv_path=args.groundtruth_csv,
        data_dir=args.data_dir,
        server_url=args.server_url
    )
    
    # Generate the map
    map_obj = plotter.run(distance_threshold=args.distance_threshold)
    
    if map_obj is not None:
        # Create and run Flask app
        app = create_flask_app(plotter)
        print(f"Starting Flask server on port {args.port}...")
        app.run(host='0.0.0.0', port=args.port, debug=False)

if __name__ == "__main__":
    main()
