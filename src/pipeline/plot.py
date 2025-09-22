import pandas as pd
import folium
import numpy as np
from geopy.distance import geodesic
import os
import sys
from typing import List, Tuple, Dict
import argparse

# config is present in root directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import Config

class TreeStreetViewPlotter:
    """Main class for plotting trees and street views."""
    
    def __init__(self):
        """Initialize the plotter."""
        self.config = Config()
        self.tree_data = None
        self.streetview_data = None
        self.cleaned_tree_data = None

    def load_data(self):
        """Load tree and street view data from CSV files."""
        print("Loading data...")
        self.tree_data = pd.read_csv(self.config.OUTPUT_CSV)
        # keep the rows where distance_pano < 12
        self.tree_data = self.tree_data[self.tree_data['distance_pano'] < 12]
        
        # Load all street view data first
        all_streetview_data = pd.read_csv(self.config.PANORAMA_CSV)
        
        # Filter street view data to only include panorama IDs that have trees
        tree_pano_ids = set(self.tree_data['pano_id'].unique())
        self.streetview_data = all_streetview_data[all_streetview_data['pano_id'].isin(tree_pano_ids)]
        
        print(f"Loaded {len(self.tree_data)} tree records")
        print(f"Loaded {len(self.streetview_data)} street view records (filtered from {len(all_streetview_data)} total)")
        print(f"Tree data covers {len(tree_pano_ids)} unique panorama IDs")

    def remove_duplicate_trees(self, distance_threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove duplicate trees based on distance threshold using efficient spatial clustering.
        
        Args:
            distance_threshold: Distance in meters to consider trees as duplicates
            
        Returns:
            DataFrame with duplicate trees removed
        """
        print(f"Removing duplicate trees within {distance_threshold}m...")
        
        # Clean data and get coordinates
        clean_data = self.tree_data[['tree_lat', 'tree_lng']].dropna()
        
        if len(clean_data) == 0:
            print("No valid tree coordinates found")
            return pd.DataFrame()
        
        # Use spatial clustering for efficient deduplication
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import pairwise_distances
        import numpy as np
        
        # Convert to numpy array for efficient processing
        coords = clean_data[['tree_lat', 'tree_lng']].values
        
        # Convert lat/lng to approximate meters for clustering
        # Rough conversion: 1 degree lat ≈ 111km, 1 degree lng ≈ 111km * cos(lat)
        lat_scale = 111000  # meters per degree latitude
        lng_scale = 111000 * np.cos(np.radians(coords[:, 0].mean()))  # meters per degree longitude
        
        # Scale coordinates to meters
        coords_scaled = coords.copy()
        coords_scaled[:, 0] *= lat_scale
        coords_scaled[:, 1] *= lng_scale
        
        # Use DBSCAN clustering to find groups within distance threshold
        clustering = DBSCAN(eps=distance_threshold, min_samples=1, metric='euclidean')
        cluster_labels = clustering.fit_predict(coords_scaled)
        
        # For each cluster, keep only the first occurrence
        unique_indices = []
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Noise points (shouldn't happen with min_samples=1)
                continue
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            # Keep the first tree in each cluster
            unique_indices.append(cluster_indices[0])
        
        # Get the original indices from clean_data
        original_indices = clean_data.index[unique_indices]
        
        # Filter the original tree_data to keep only unique trees
        self.cleaned_tree_data = self.tree_data.loc[original_indices].copy()
        
        removed_count = len(self.tree_data) - len(self.cleaned_tree_data)
        print(f"Removed {removed_count} duplicate trees, kept {len(self.cleaned_tree_data)} unique trees")
        
        return self.cleaned_tree_data
    
    def create_popup(self, title: str) -> str:
        """
        Create simple text popup content.
        
        Args:
            title: Title for the popup
            
        Returns:
            HTML string for popup
        """
        return f"<div style='text-align: center;'><h4>{title}</h4></div>"
    
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
                # Create simple popup
                popup_html = self.create_popup(f"Tree at ({row['tree_lat']:.6f}, {row['tree_lng']:.6f})")
                
                folium.CircleMarker(
                    [row['tree_lat'], row['tree_lng']],
                    popup=folium.Popup(popup_html, max_width=200),
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
                # Create simple popup
                popup_html = self.create_popup(f"Street View at ({row['lat']:.6f}, {row['lng']:.6f})")
                
                folium.CircleMarker(
                    [row['lat'], row['lng']],
                    popup=folium.Popup(popup_html, max_width=200),
                    radius=0.2,
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.8
                ).add_to(m)
        
        # Add connection lines between trees and street views
        print("Adding connection lines...")
        self._add_connection_lines(m)
        
        # Add legend
        self._add_legend(m)
        
        return m
    
    def _add_connection_lines(self, map_obj: folium.Map):
        """Add lines connecting trees to their corresponding street views (edge-to-edge)."""
        # Group trees by pano_id
        tree_groups = self.cleaned_tree_data.groupby('pano_id')
        
        for pano_id, tree_group in tree_groups:
            # Find corresponding street view
            streetview = self.streetview_data[self.streetview_data['pano_id'] == pano_id]
            
            if len(streetview) > 0:
                sv_lat = streetview.iloc[0]['lat']
                sv_lng = streetview.iloc[0]['lng']
                
                # Draw lines from street view to each tree (edge-to-edge)
                for _, tree in tree_group.iterrows():
                    if pd.notna(tree['tree_lat']) and pd.notna(tree['tree_lng']):
                        # Calculate edge-to-edge connection
                        start_point, end_point = self._calculate_edge_points(
                            sv_lat, sv_lng, tree['tree_lat'], tree['tree_lng']
                        )
                        
                        folium.PolyLine(
                            locations=[start_point, end_point],
                            color='red',
                            weight=1,
                            opacity=0.6,
                            popup=f"Connection: {pano_id}"
                        ).add_to(map_obj)
    
    def _calculate_edge_points(self, lat1: float, lng1: float, lat2: float, lng2: float):
        """Calculate edge-to-edge connection points between two markers."""
        # Marker radius in degrees (approximate conversion from 0.2 radius)
        marker_radius_deg = 0.000002  # Rough conversion for 0.2 radius
        
        # Calculate bearing (direction) from point 1 to point 2
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        d_lng = np.radians(lng2 - lng1)
        
        y = np.sin(d_lng) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(d_lng)
        bearing = np.arctan2(y, x)
        
        # Calculate edge points
        # For point 1 (street view), extend towards point 2
        start_lat = lat1 + marker_radius_deg * np.cos(bearing)
        start_lng = lng1 + marker_radius_deg * np.sin(bearing) / np.cos(np.radians(lat1))
        
        # For point 2 (tree), extend towards point 1
        end_lat = lat2 - marker_radius_deg * np.cos(bearing)
        end_lng = lng2 - marker_radius_deg * np.sin(bearing) / np.cos(np.radians(lat2))
        
        return [start_lat, start_lng], [end_lat, end_lng]
    
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
    
    def save_map(self, filename: str = None):
        """Save the map to an HTML file."""
        if filename is None:
            filename = f"tree_map.html"
        
        # Ensure filename has .html extension
        if not filename.endswith('.html'):
            filename += '.html'
        
        # Save the map
        self.map_obj.save(filename)
        print(f"Map saved to: {filename}")
        return filename
    
    def run(self, distance_threshold: float = 3.0, save_map: bool = True):
        """
        Run the complete plotting pipeline.
        
        Args:
            distance_threshold: Distance threshold for duplicate removal in meters
            save_map: Whether to save the map to HTML file
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
            
            # Save map
            if save_map:
                saved_file = self.save_map()
                print(f"Map saved to: {saved_file}")
            
            print(f"\nMap created successfully!")
            
            return self.map_obj
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

def main():
    """Main function to run the plotter from command line."""
    parser = argparse.ArgumentParser(description='Plot trees and street views on interactive map')
    parser.add_argument('--distance-threshold', type=float, default=3.0, 
                       help='Distance threshold for duplicate removal in meters')
    parser.add_argument('--no-save', action='store_true', help='Do not save map to HTML file')
    
    args = parser.parse_args()
    
    # Create and run plotter
    plotter = TreeStreetViewPlotter()
    
    # Generate the map
    map_obj = plotter.run(
        distance_threshold=args.distance_threshold,
        save_map=not args.no_save
    )
    
    if map_obj is not None:
        print("Map generation completed successfully!")

if __name__ == "__main__":
    main()
