import pandas as pd
import pydeck as pdk
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
    """Main class for plotting trees and street views using Deck.gl."""
    
    def __init__(self, show_connections: bool = True):
        """Initialize the plotter.
        
        Args:
            show_connections: Whether to show connection lines between trees and street views
        """
        self.config = Config()
        self.tree_data = None
        self.streetview_data = None
        self.cleaned_tree_data = None
        self.show_connections = show_connections
        self.deck = None

    def load_data(self):
        """Load tree and street view data from CSV files."""
        print("Loading data...")
        self.tree_data = pd.read_csv(self.config.OUTPUT_CSV)
        # keep the rows where distance_pano < 12
        self.tree_data = self.tree_data[self.tree_data['distance_pano'] < 12]
        
        # Load all street view data
        self.streetview_data = pd.read_csv(self.config.PANORAMA_CSV)
        
        print(f"Loaded {len(self.tree_data)} tree records")
        print(f"Loaded {len(self.streetview_data)} street view records")
        print(f"Tree data covers {len(set(self.tree_data['pano_id'].unique()))} unique panorama IDs")

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
    
    def _prepare_tree_data(self) -> pd.DataFrame:
        """Prepare tree data for Deck.gl ScatterplotLayer - optimized."""
        print("Preparing tree data...")
        
        # Filter valid coordinates
        valid_trees = self.cleaned_tree_data.dropna(subset=['tree_lat', 'tree_lng']).copy()
        
        # Add color and size for trees using vectorized operations
        valid_trees['color'] = [[34, 139, 34, 200]] * len(valid_trees)  # Forest green with transparency
        valid_trees['radius'] = 1  # Small tree marker size
        
        # Select and rename columns efficiently
        result = valid_trees[['tree_lng', 'tree_lat', 'color', 'radius', 'pano_id']].rename(columns={
            'tree_lng': 'lon', 
            'tree_lat': 'lat'
        })
        
        print(f"Prepared {len(result)} tree markers")
        return result
    
    def _prepare_streetview_data(self) -> pd.DataFrame:
        """Prepare street view data for Deck.gl ScatterplotLayer - optimized."""
        print("Preparing street view data...")
        
        # Filter valid coordinates
        valid_streetviews = self.streetview_data.dropna(subset=['lat', 'lng']).copy()
        
        # Add color and size for street views using vectorized operations
        valid_streetviews['color'] = [[30, 144, 255, 200]] * len(valid_streetviews)  # Dodger blue with transparency
        valid_streetviews['radius'] = 1  # Very small street view marker size
        
        # Select and rename columns efficiently
        result = valid_streetviews[['lng', 'lat', 'color', 'radius', 'pano_id']].rename(columns={
            'lng': 'lon'
        })
        
        print(f"Prepared {len(result)} street view markers")
        return result
    
    def _prepare_connection_data(self) -> pd.DataFrame:
        """Prepare connection data for Deck.gl LineLayer - optimized for large datasets."""
        if not self.show_connections:
            return pd.DataFrame()
        
        print("Preparing connection data (this may take a moment for large datasets)...")
        
        # Use pandas merge for efficient joining instead of loops
        # First, get streetview coordinates for each pano_id
        sv_coords = self.streetview_data[['pano_id', 'lat', 'lng']].rename(columns={
            'lat': 'source_lat', 
            'lng': 'source_lon'
        })
        
        # Merge tree data with streetview coordinates
        tree_with_sv = self.cleaned_tree_data.merge(sv_coords, on='pano_id', how='inner')
        
        # Filter out rows with missing coordinates
        valid_connections = tree_with_sv.dropna(subset=['tree_lat', 'tree_lng', 'source_lat', 'source_lon'])
        
        # Select and rename columns for Deck.gl LineLayer
        connections_df = valid_connections[[
            'source_lon', 'source_lat', 'tree_lng', 'tree_lat', 'pano_id'
        ]].rename(columns={
            'tree_lng': 'target_lon',
            'tree_lat': 'target_lat'
        })
        
        print(f"Prepared {len(connections_df)} connection lines")
        return connections_df
    
    def create_map(self) -> pdk.Deck:
        """
        Create the interactive map with trees, street views, and connections using Deck.gl.
        
        Returns:
            pdk.Deck object
        """
        print("Creating Deck.gl map...")
        
        # Prepare data for each layer with progress tracking
        print("Step 1/4: Preparing data layers...")
        tree_data = self._prepare_tree_data()
        streetview_data = self._prepare_streetview_data()
        connection_data = self._prepare_connection_data()
        
        print("Step 2/4: Calculating map bounds and zoom...")
        
        # Calculate center point and bounds
        all_lats = list(tree_data['lat']) + list(streetview_data['lat'])
        all_lons = list(tree_data['lon']) + list(streetview_data['lon'])
        
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        # Calculate zoom level based on data bounds
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        max_range = max(lat_range, lon_range)
        
        # Estimate zoom level (rough approximation)
        if max_range > 1:
            zoom = 8
        elif max_range > 0.1:
            zoom = 11
        elif max_range > 0.01:
            zoom = 14
        else:
            zoom = 16
        
        print(f"Map center: ({center_lat:.6f}, {center_lon:.6f}), zoom: {zoom}")
        
        # Store for use in save_map
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        
        print("Step 3/4: Creating Deck.gl layers...")
        
        # Define layers with proper visibility controls
        layers = []
        
        # OpenStreetMap base layer (more reliable)
        # tile_layer = pdk.Layer(
        #     "TileLayer",
        #     data={
        #         "tileSize": 256,
        #         "minZoom": 0,
        #         "maxZoom": 19,
        #         "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        #     },
        # )
        # layers.append(tile_layer)
        esri_layer = pdk.Layer(
            "TileLayer",
            data={
                "tileSize": 256,
                "minZoom": 0,
                "maxZoom": 19,
                "tiles": [
                    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
            },
            id="esri-imagery",
        )
        layers.append(esri_layer)
        
        # Connection lines layer (drawn first, so it appears below markers)
        if self.show_connections and len(connection_data) > 0:
            connection_layer = pdk.Layer(
                "LineLayer",
                connection_data,
                id="connections",
                get_source_position=["source_lon", "source_lat"],
                get_target_position=["target_lon", "target_lat"],
                get_color=[255, 69, 0, 100],  # Orange red with transparency
                get_width=0.5,  # Thinner lines
                pickable=True,
                auto_highlight=True,
                visible=True
            )
            layers.append(connection_layer)
        
        # Street view markers layer
        if len(streetview_data) > 0:
            streetview_layer = pdk.Layer(
                "ScatterplotLayer",
                streetview_data,
                id="streetviews",
                get_position=["lon", "lat"],
                get_color="color",
                get_radius="radius",
                radius_min_pixels=1,
                radius_max_pixels=10,
                pickable=True,
                auto_highlight=True,
                visible=True
            )
            layers.append(streetview_layer)
        
        # Tree markers layer (drawn last, so it appears on top)
        if len(tree_data) > 0:
            tree_layer = pdk.Layer(
                "ScatterplotLayer",
                tree_data,
                id="trees",
                get_position=["lon", "lat"],
                get_color="color",
                get_radius="radius",
                radius_min_pixels=1,
                radius_max_pixels=15,
                pickable=True,
                auto_highlight=True,
                visible=True
            )
            layers.append(tree_layer)
        
        print("Step 4/4: Assembling final Deck.gl map...")
        
        # Create the deck
        self.deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom,
                pitch=0,
                bearing=0
            ),
            map_provider="carto",  # or "mapbox"
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={
                "html": """
                <b>Location:</b> {lat:.6f}, {lon:.6f}<br/>
                <b>Panorama ID:</b> {pano_id}
                """,
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white",
                    "fontSize": "12px",
                    "padding": "8px",
                    "borderRadius": "4px"
                }
            }
        )
        
        print("Deck.gl map created successfully")
        return self.deck

    def save_map(self, filename: str = None):
        """Save the map to an HTML file with custom layer + basemap controls."""
        try:
            if filename is None:
                filename = "tree_map_deckgl.html"
            
            if not filename.endswith('.html'):
                filename += '.html'
            
            print(f"Saving Deck.gl map to {filename}...")
            
            # Save base map
            self.deck.to_html(filename, open_browser=False)
            
            # Read HTML
            with open(filename, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"Map saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving map: {e}")
            raise
    

    def run(self, distance_threshold: float = 3.0, save_map: bool = True, show_connections: bool = None):
        """
        Run the complete plotting pipeline.
        
        Args:
            distance_threshold: Distance threshold for duplicate removal in meters
            save_map: Whether to save the map to HTML file
            show_connections: Whether to show connection lines (overrides instance setting)
        """
        # Update show_connections if provided
        if show_connections is not None:
            self.show_connections = show_connections
            
        try:
            # Load data
            self.load_data()
            
            # Remove duplicates
            self.cleaned_tree_data = self.remove_duplicate_trees(distance_threshold)
            
            if len(self.cleaned_tree_data) == 0:
                print("No tree data to plot after duplicate removal")
                return None
            
            # Create map
            print("Starting Deck.gl map creation...")
            deck_map = self.create_map()
            print("Deck.gl map object created successfully")
            
            # Save map
            if save_map:
                print("Starting map save...")
                saved_file = self.save_map()
                print(f"Map saved to: {saved_file}")
            
            print(f"\nDeck.gl map created successfully!")
            print(f"Total trees displayed: {len(self.cleaned_tree_data)}")
            print(f"Total street views displayed: {len(self.streetview_data)}")
            if self.show_connections:
                connection_count = len(self._prepare_connection_data())
                print(f"Total connections displayed: {connection_count}")
            
            return deck_map
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

def main():
    """Main function to run the plotter from command line."""
    parser = argparse.ArgumentParser(description='Plot trees and street views on interactive Deck.gl map')
    parser.add_argument('--distance-threshold', type=float, default=3.0, 
                       help='Distance threshold for duplicate removal in meters')
    parser.add_argument('--no-save', action='store_true', help='Do not save map to HTML file')
    parser.add_argument('--no-connections', action='store_true', 
                       help='Do not show connection lines between trees and street views')
    
    args = parser.parse_args()
    
    # Create and run plotter
    plotter = TreeStreetViewPlotter(
        show_connections=not args.no_connections
    )
    
    # Generate the map
    map_obj = plotter.run(
        distance_threshold=args.distance_threshold,
        save_map=not args.no_save
    )
    
    if map_obj is not None:
        print("Deck.gl map generation completed successfully!")

if __name__ == "__main__":
    main()