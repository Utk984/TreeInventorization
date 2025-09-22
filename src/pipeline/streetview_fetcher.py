#!/usr/bin/env python3
"""
Fast Street View Fetcher for GeoJSON Polygons
Fetches street view panoramas from coordinates within GeoJSON polygons
and saves them as CSV with lat, lng, pano_id format.
"""

import json
import csv
import asyncio
import aiohttp
from typing import List, Tuple, Optional, Set
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import sys
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

try:
    from streetlevel import streetview
except ImportError:
    print("Installing streetlevel library...")
    import subprocess
    subprocess.check_call(["pip", "install", "streetlevel"])
    from streetlevel import streetview

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreetViewFetcher:
    """Fast street view fetcher for GeoJSON polygons."""
    
    def __init__(self, max_workers: int = 100, radius: int = 50):
        """
        Initialize the fetcher.
        
        Args:
            max_workers: Maximum number of concurrent workers
            radius: Search radius in meters for finding panoramas
        """
        self.max_workers = max_workers
        self.radius = radius
        self.results = []
        self.completed = 0
        self.total = 0
        self.successful = 0
        self.failed = 0
        self.failure_reasons = {}
        
    def extract_coordinates_from_geojson(self, geojson_path: str, sample_density: float = 0.001) -> List[Tuple[float, float]]:
        """
        Extract coordinates from GeoJSON polygons.
        
        Args:
            geojson_path: Path to GeoJSON file
            sample_density: Density of coordinate sampling (smaller = more points)
            
        Returns:
            List of (lat, lng) tuples
        """
        print("📁 Reading GeoJSON file...")
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        coordinates = []
        features = geojson_data['features']
        total_features = len(features)
        
        print(f"🗺️  Processing {total_features} features from GeoJSON...")
        
        for i, feature in enumerate(features):
            feature_name = feature.get('properties', {}).get('Ward_Name', f'Feature {i+1}')
            print(f"   📍 Processing {feature_name}...", end=' ', flush=True)
            
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]  # Exterior ring
                
                # Sample coordinates from the polygon
                sampled_coords = self._sample_polygon_coordinates(coords, sample_density)
                coordinates.extend(sampled_coords)
                print(f"✅ {len(sampled_coords)} points")
                
            elif feature['geometry']['type'] == 'MultiPolygon':
                total_polygon_coords = 0
                for polygon in feature['geometry']['coordinates']:
                    coords = polygon[0]  # Exterior ring
                    sampled_coords = self._sample_polygon_coordinates(coords, sample_density)
                    coordinates.extend(sampled_coords)
                    total_polygon_coords += len(sampled_coords)
                print(f"✅ {total_polygon_coords} points")
        
        print(f"🎯 Total coordinates extracted: {len(coordinates)}")
        return coordinates
    
    def _sample_polygon_coordinates(self, coords: List[List[float]], density: float) -> List[Tuple[float, float]]:
        """Sample coordinates from a polygon with given density.
        
        For coverage tiles (150x150m), density of 0.00135 degrees ≈ 150m spacing.
        This ensures each sampled point corresponds to a unique coverage tile.
        """
        if len(coords) < 3:
            return []
        
        # Get bounding box
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Generate grid points with tile-optimized spacing
        sampled = []
        lon_step = (max_lon - min_lon) * density
        lat_step = (max_lat - min_lat) * density
        
        lon = min_lon
        while lon <= max_lon:
            lat = min_lat
            while lat <= max_lat:
                # Check if point is inside polygon (simple ray casting)
                if self._point_in_polygon(lon, lat, coords):
                    sampled.append((lat, lon))  # Note: lat, lon order for streetview
                lat += lat_step
            lon += lon_step
        
        return sampled
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[List[float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _update_progress(self, success: bool = True, count: int = 1, failure_reason: str = None):
        """Update progress counters and display progress bar."""
        self.completed += 1
        if success:
            self.successful += count
        else:
            self.failed += 1
            if failure_reason:
                self.failure_reasons[failure_reason] = self.failure_reasons.get(failure_reason, 0) + 1
        
        # Calculate progress percentage
        progress = (self.completed / self.total) * 100 if self.total > 0 else 0
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * self.completed // self.total) if self.total > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Display progress
        print(f'\r🔄 Progress: |{bar}| {progress:.1f}% ({self.completed}/{self.total}) ✅ {self.successful} panoramas ❌ {self.failed} failed', end='', flush=True)
        
        # New line when complete
        if self.completed >= self.total:
            print()  # New line when done
    
    async def fetch_coverage_tile_async(self, session: aiohttp.ClientSession, lat: float, lon: float) -> List[dict]:
        """Fetch multiple panoramas from a coverage tile asynchronously."""
        try:
            # Get coverage tile which returns multiple panoramas
            panoramas = await streetview.get_coverage_tile_by_latlon_async(lat, lon, session)
            
            if panoramas:
                results = []
                for pano in panoramas:
                    results.append({
                        'lat': pano.lat,
                        'lng': pano.lon,
                        'pano_id': pano.id
                    })
                self._update_progress(success=True, count=len(results))
                return results
            else:
                self._update_progress(success=False, failure_reason="No panoramas found")
                return []
        except aiohttp.ClientError as e:
            error_msg = f"Network error: {str(e)[:50]}"
            logger.debug(f"Network error for ({lat}, {lon}): {e}")
            self._update_progress(success=False, failure_reason=error_msg)
            return []
        except asyncio.TimeoutError as e:
            error_msg = "Request timeout"
            logger.debug(f"Timeout for ({lat}, {lon}): {e}")
            self._update_progress(success=False, failure_reason=error_msg)
            return []
        except Exception as e:
            error_msg = f"Unknown error: {str(e)[:50]}"
            logger.debug(f"Failed to fetch coverage tile for ({lat}, {lon}): {e}")
            self._update_progress(success=False, failure_reason=error_msg)
            return []
    
    async def fetch_all_panoramas(self, coordinates: List[Tuple[float, float]], output_path: str) -> List[dict]:
        """Fetch panoramas for all coordinates asynchronously using ULTRA-FAST parallel processing."""
        # Initialize progress tracking
        self.total = len(coordinates)
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.failure_reasons = {}
        
        print(f"\n🚀 Starting ULTRA-FAST fetch with {self.max_workers} workers...")
        print(f"📊 Processing {self.total} coverage tiles in parallel...")
        print("💡 Maximum concurrency enabled!")
        print("💾 Saving progress every 1000 panoramas...")
        
        # Initialize CSV file
        if Path(output_path).exists():
            Path(output_path).unlink()  # Remove existing file
        
        all_panoramas = []
        batch_size = 1000  # Save every 1000 panoramas
        
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=200,           # Total connection pool size
                limit_per_host=100, # Connections per host
                ttl_dns_cache=300,  # DNS cache
                use_dns_cache=True
            ),
            timeout=aiohttp.ClientTimeout(
                total=30,           # Total timeout
                connect=10,         # Connection timeout
                sock_read=10        # Socket read timeout
            )
        ) as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def fetch_with_semaphore(lat, lon):
                async with semaphore:
                    return await self.fetch_coverage_tile_async(session, lat, lon)
            
            # Create ALL tasks at once for maximum parallelism
            print("🔥 Creating all tasks for maximum parallelism...")
            tasks = [fetch_with_semaphore(lat, lon) for lat, lon in coordinates]
            
            # Execute ALL tasks concurrently
            print("⚡ Executing ALL tasks concurrently...")
            start_time = time.time()
            
            # Process in chunks to enable incremental saving
            chunk_size = self.max_workers * 10  # Process 10x max_workers at a time
            
            for i in range(0, len(tasks), chunk_size):
                chunk_tasks = tasks[i:i + chunk_size]
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                
                # Process chunk results
                for result in chunk_results:
                    if isinstance(result, list):
                        all_panoramas.extend(result)
                    elif isinstance(result, dict):
                        all_panoramas.append(result)
                
                # Save incrementally if we have enough panoramas
                if len(all_panoramas) >= batch_size:
                    remaining = self.save_incremental_csv(all_panoramas, output_path, batch_size)
                    all_panoramas = remaining
            
            end_time = time.time()
            print(f"\n⏱️  Completed in {end_time - start_time:.2f} seconds")
        
        # Save any remaining panoramas
        if all_panoramas:
            self.save_incremental_csv(all_panoramas, output_path, 1)  # Save all remaining
        
        # Final deduplication and statistics
        seen_pano_ids = set()
        unique_panoramas = []
        duplicates_removed = 0
        
        for pano in all_panoramas:
            pano_id = pano.get('pano_id')
            if pano_id and pano_id not in seen_pano_ids:
                seen_pano_ids.add(pano_id)
                unique_panoramas.append(pano)
            else:
                duplicates_removed += 1
        
        print(f"📈 Final Results:")
        print(f"   ✅ Total Panoramas Found: {len(all_panoramas)}")
        print(f"   🔄 Duplicates Removed: {duplicates_removed}")
        print(f"   🎯 Unique Panoramas: {len(unique_panoramas)}")
        print(f"   🗺️  Coverage Tiles Processed: {self.completed}")
        print(f"   ❌ Failed Tiles: {self.failed}")
        print(f"   📊 Success Rate: {((self.completed - self.failed)/self.completed)*100:.1f}%")
        
        # Print failure reasons
        if self.failure_reasons:
            print(f"🔍 Failure Analysis:")
            for reason, count in self.failure_reasons.items():
                print(f"   • {reason}: {count} failures")
        
        return unique_panoramas
    
    def save_to_csv(self, results: List[dict], output_path: str, append: bool = False):
        """Save results to CSV file."""
        mode = 'a' if append else 'w'
        write_header = not append or not Path(output_path).exists()
        
        with open(output_path, mode, newline='', encoding='utf-8') as csvfile:
            fieldnames = ['lat', 'lng', 'pano_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            
            for result in results:
                writer.writerow(result)
    
    def save_incremental_csv(self, all_panoramas: List[dict], output_path: str, batch_size: int = 100):
        """Save panoramas incrementally to CSV to prevent data loss."""
        # Remove duplicates before saving
        seen_pano_ids = set()
        unique_panoramas = []
        
        for pano in all_panoramas:
            pano_id = pano.get('pano_id')
            if pano_id and pano_id not in seen_pano_ids:
                seen_pano_ids.add(pano_id)
                unique_panoramas.append(pano)
        
        # Save in batches
        if len(unique_panoramas) >= batch_size:
            batch = unique_panoramas[:batch_size]
            remaining = unique_panoramas[batch_size:]
            
            # Save batch
            self.save_to_csv(batch, output_path, append=True)
            
            # Return remaining for further processing
            return remaining
        
        return unique_panoramas
    
    async def process_geojson(self, geojson_path: str, output_path: str, sample_density: float = 0.001):
        """Process GeoJSON file and save results to CSV."""
        print(f"\n{'='*60}")
        print(f"🚀 STREET VIEW FETCHER STARTED")
        print(f"{'='*60}")
        print(f"📁 Input: {geojson_path}")
        print(f"📁 Output: {output_path}")
        print(f"⚙️  Settings: {self.max_workers} workers, {self.radius}m radius, density {sample_density}")
        print(f"{'='*60}\n")
        
        # Extract coordinates
        print("🔍 PHASE 1: Extracting coordinates from GeoJSON...")
        coordinates = self.extract_coordinates_from_geojson(geojson_path, sample_density)
        # reverse the coordinates
        
        if not coordinates:
            print("❌ No coordinates found in GeoJSON")
            return
        
        print(f"\n🌐 PHASE 2: Fetching street view panoramas...")
        # Fetch panoramas (with incremental saving)
        results = await self.fetch_all_panoramas(coordinates, output_path)
        
        if not results:
            print("❌ No panoramas found")
            return
        
        print(f"\n✅ PHASE 3: Results already saved incrementally to {output_path}")
        
        print(f"\n{'='*60}")
        print(f"🎉 STREET VIEW FETCH COMPLETED!")
        print(f"📊 Final Results: {len(results)} panoramas saved to {output_path}")
        print(f"{'='*60}")


class NeighborDiscovery:
    """Fast recursive neighbor discovery with maximum threading."""
    
    def __init__(self, geojson_path: str = None, max_workers: int = 100):
        """
        Initialize the neighbor discovery system.
        
        Args:
            geojson_path: Path to GeoJSON file for boundary checking
            max_workers: Maximum number of concurrent workers for speed
        """
        self.geojson_path = geojson_path
        self.max_workers = max_workers
        self.boundary_polygons = []
        self.prepared_polygons = []
        self.all_pano_ids = set()  # Track all pano_ids we've seen
        
    def load_panoramas(self, csv_path: str) -> List[dict]:
        """Load panoramas from CSV file."""
        print(f"📁 Loading panoramas from {csv_path}...")
        
        panoramas = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pano_id = row['pano_id']
                lat = float(row['lat'])
                lng = float(row['lng'])
                
                panoramas.append({
                    'lat': lat,
                    'lng': lng,
                    'pano_id': pano_id
                })
                self.all_pano_ids.add(pano_id)
        
        print(f"✅ Loaded {len(panoramas)} panoramas")
        return panoramas
    
    def load_boundary_polygons(self):
        """Load and prepare GeoJSON boundary polygons for fast point-in-polygon checks."""
        if not self.geojson_path or not Path(self.geojson_path).exists():
            print("⚠️  No GeoJSON boundary file provided - skipping boundary checks")
            return
        
        print(f"🗺️  Loading boundary polygons from {self.geojson_path}...")
        
        with open(self.geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]  # Exterior ring
                polygon = Polygon(coords)
                self.boundary_polygons.append(polygon)
                self.prepared_polygons.append(prep(polygon))
            elif feature['geometry']['type'] == 'MultiPolygon':
                for polygon_coords in feature['geometry']['coordinates']:
                    coords = polygon_coords[0]  # Exterior ring
                    polygon = Polygon(coords)
                    self.boundary_polygons.append(polygon)
                    self.prepared_polygons.append(prep(polygon))
        
        print(f"✅ Loaded {len(self.boundary_polygons)} boundary polygons")
    
    def is_within_boundary(self, lat: float, lng: float) -> bool:
        """Check if a point is within the boundary polygons."""
        if not self.prepared_polygons:
            return True  # No boundary restrictions
        
        point = Point(lng, lat)  # Note: Point takes (x, y) = (lng, lat)
        
        for prepared_polygon in self.prepared_polygons:
            if prepared_polygon.contains(point):
                return True
        
        return False
    
    async def fetch_neighbors_async(self, session: aiohttp.ClientSession, pano_id: str) -> List[dict]:
        """Fetch neighbors for a given panorama ID asynchronously."""
        try:
            # Get panorama details first to access neighbors
            pano = await streetview.find_panorama_by_id_async(pano_id, session, download_depth=False)
            if not pano or not pano.neighbors:
                return []
            
            valid_neighbors = []
            for neighbor in pano.neighbors:
                neighbor_id = neighbor.id
                neighbor_lat = neighbor.lat
                neighbor_lng = neighbor.lon
                
                # Skip if already exists
                if neighbor_id in self.all_pano_ids:
                    continue
                
                # Check if within boundary
                if not self.is_within_boundary(neighbor_lat, neighbor_lng):
                    continue
                
                valid_neighbors.append({
                    'lat': neighbor_lat,
                    'lng': neighbor_lng,
                    'pano_id': neighbor_id
                })
                self.all_pano_ids.add(neighbor_id)
            
            return valid_neighbors
            
        except Exception as e:
            logger.debug(f"Failed to fetch neighbors for {pano_id}: {e}")
            return []
    
    def save_to_csv(self, panoramas: List[dict], output_path: str):
        """Save panoramas to CSV file."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['lat', 'lng', 'pano_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(panoramas)
    

    async def discover_neighbors(self, csv_path: str, output_path: str = None) -> List[dict]:
        """Ultra-fast recursive neighbor discovery - only processes NEW panoramas each iteration."""
        print(f"\n{'='*60}")
        print(f"🚀 ULTRA-FAST RECURSIVE NEIGHBOR DISCOVERY")
        print(f"{'='*60}")
        
        # Load existing panoramas
        panoramas = self.load_panoramas(csv_path)
        
        # Load boundary polygons
        self.load_boundary_polygons()
        
        # Set output path
        if output_path is None:
            output_path = csv_path.replace('.csv', '_with_neighbors.csv')
        
        print(f"🚀 Starting with {len(panoramas)} panoramas...")
        print(f"⚡ Using {self.max_workers} concurrent workers for maximum speed...")
        
        # Process panoramas until no new neighbors are found
        all_panoramas = panoramas.copy()
        processed_pano_ids = set()  # Track which pano_ids we've already processed
        iteration = 0
        
        # Add initial panoramas to processed set
        for pano in panoramas:
            processed_pano_ids.add(pano['pano_id'])
        
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=200,           # High connection pool
                limit_per_host=100, # High per-host limit
                ttl_dns_cache=300,
                use_dns_cache=True
            ),
            timeout=aiohttp.ClientTimeout(
                total=30,
                connect=10,
                sock_read=10
            )
        ) as session:
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def fetch_with_semaphore(pano_id):
                async with semaphore:
                    return await self.fetch_neighbors_async(session, pano_id)
            
            while True:
                iteration += 1
                
                # Get ONLY NEW panoramas to process in this iteration
                new_panoramas = [pano for pano in all_panoramas if pano['pano_id'] not in processed_pano_ids]
                
                if not new_panoramas:
                    print(f"   🎯 No new panoramas to process - stopping recursion")
                    break
                
                print(f"\n🔄 Iteration {iteration} - Processing {len(new_panoramas)} NEW panoramas...")
                print(f"   📊 Total panoramas so far: {len(all_panoramas)}")
                
                new_neighbors_found = 0
                
                # Process in parallel batches for maximum speed
                batch_size = self.max_workers * 2
                
                for i in range(0, len(new_panoramas), batch_size):
                    batch = new_panoramas[i:i + batch_size]
                    
                    # Create tasks for this batch
                    tasks = [fetch_with_semaphore(pano['pano_id']) for pano in batch]
                    
                    # Execute batch in parallel
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for j, result in enumerate(batch_results):
                        pano_id = batch[j]['pano_id']
                        processed_pano_ids.add(pano_id)  # Mark as processed
                        
                        if isinstance(result, list):
                            for neighbor in result:
                                all_panoramas.append(neighbor)
                                new_neighbors_found += 1
                        elif isinstance(result, Exception):
                            logger.debug(f"Task failed for {pano_id}: {result}")
                    
                    # Show progress
                    if (i + batch_size) % (batch_size * 5) == 0 or i + batch_size >= len(new_panoramas):
                        print(f"   🔄 Processed {min(i + batch_size, len(new_panoramas))}/{len(new_panoramas)} | Total: {len(all_panoramas)}")
                
                print(f"   ✅ Found {new_neighbors_found} new neighbors in iteration {iteration}")
                
                # If no new neighbors found, we're done
                if new_neighbors_found == 0:
                    print(f"   🎯 No new neighbors found - stopping recursion")
                    break
        
        # Save all panoramas to CSV
        print(f"\n💾 Saving {len(all_panoramas)} panoramas to {output_path}...")
        self.save_to_csv(all_panoramas, output_path)
        
        # Final statistics
        total_new_neighbors = len(all_panoramas) - len(panoramas)
        print(f"\n📈 Results:")
        print(f"   🎯 Original Panoramas: {len(panoramas)}")
        print(f"   🔍 Total New Neighbors Found: {total_new_neighbors}")
        print(f"   📊 Total Panoramas: {len(all_panoramas)}")
        print(f"   🔄 Iterations: {iteration}")
        print(f"   📁 Output File: {output_path}")
        
        return all_panoramas


async def main_neighbor_discovery():
    """Ultra-fast recursive neighbor discovery."""
    
    # File paths
    csv_path = "streetviews/south_delhi.csv"
    geojson_path = "streetviews/South Delhi.geojson"
    output_path = "streetviews/south_delhi_with_neighbors.csv"
    
    # Check if CSV exists
    if not Path(csv_path).exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return
    
    # Create ultra-fast neighbor discovery
    discovery = NeighborDiscovery(geojson_path=geojson_path, max_workers=100)
    
    try:
        # Discover neighbors recursively
        all_panoramas = await discovery.discover_neighbors(
            csv_path=csv_path,
            output_path=output_path
        )
        
        print(f"\n{'='*60}")
        print(f"🎉 ULTRA-FAST NEIGHBOR DISCOVERY COMPLETED!")
        print(f"📊 Total panoramas: {len(all_panoramas)}")
        print(f"📁 Results saved to {output_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

async def main():
    """Fetch street views for South Delhi."""
    
    # File paths
    geojson_path = "streetviews/South Delhi.geojson"
    output_path = "streetviews/south_delhi_streets.csv"
    
    # Check if GeoJSON exists
    if not Path(geojson_path).exists():
        print(f"❌ Error: GeoJSON file not found at {geojson_path}")
        return
    
    # Create fetcher with ULTRA-FAST settings
    # Using maximum workers for fastest processing
    fetcher = StreetViewFetcher(max_workers=100, radius=50)
    
    try:
        # Process the GeoJSON with ULTRA-FAST sampling density for coverage tiles
        # Coverage tiles are 150x150 meters, so we sample every ~300m for speed
        await fetcher.process_geojson(
            geojson_path=geojson_path,
            output_path=output_path,
            sample_density=0.0001  # ~150 meters between samples (matches tile size)
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "neighbors":
        print("🚀 Running Ultra-Fast Recursive Neighbor Discovery...")
        asyncio.run(main_neighbor_discovery())
    else:
        print("🚀 Running Street View Fetch Mode...")
        asyncio.run(main())

