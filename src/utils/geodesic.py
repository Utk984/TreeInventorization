import math
import logging
import time
import numpy as np
from dataclasses import dataclass

# Configure logger for geodesic calculations
logger = logging.getLogger(__name__)

def get_coordinates(pano, point, width, distance):
    """
    Calculate GPS coordinates from panorama position and detected tree location with logging.
    
    Args:
        pano: Panorama object with lat, lon, and heading
        point: (x, y) pixel coordinates of tree in equirectangular image
        width: Width of the equirectangular image
        distance: Distance to tree in meters
        
    Returns:
        tuple: (latitude, longitude) of the tree
    """
    logger.debug(f"🌍 Calculating GPS coordinates for point {point} at distance {distance:.2f}m")
    start_time = time.time()
    
    try:
        center = width / 2
        bearing_radians = pano.heading / center * point[0]
        R = 6371000  # Earth radius in meters

        # Calculate angular distance in radians
        angular_distance = distance / R
        
        # Convert latitude and longitude from degrees to radians
        lat_rad = math.radians(pano.lat)
        lng_rad = math.radians(pano.lon)
        
        logger.debug(f"📍 Panorama location: ({pano.lat:.6f}, {pano.lon:.6f})")
        logger.debug(f"🧭 Bearing: {math.degrees(bearing_radians):.2f}°, Angular distance: {angular_distance:.6f} rad")
        
        # Calculate new latitude using spherical trigonometry
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance) +
            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_radians)
        )
        
        # Calculate new longitude
        new_lng_rad = lng_rad + math.atan2(
            math.sin(bearing_radians) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        # Convert back to degrees
        lat = math.degrees(new_lat_rad)
        lng = math.degrees(new_lng_rad)
        
        calc_time = time.time() - start_time
        logger.debug(f"✅ GPS calculation completed in {calc_time:.3f}s")
        logger.debug(f"📍 Tree coordinates: ({lat:.6f}, {lng:.6f})")
        
        return lat, lng
        
    except Exception as e:
        logger.error(f"❌ Error calculating GPS coordinates: {str(e)}")
        logger.error(f"📊 Input parameters: point={point}, width={width}, distance={distance}")
        raise

import numpy as np

def get_depth_at_pixel(depth_map, u, v, W, H, flipped=True, method="nearest"):
    """
    Return the depth (meters) at high-res pixel (u, v) from a low-res Street View depth map.
    
    Args:
        depth_map: streetlevel DepthMap object with .data as a NumPy array (shape: H_d x W_d).
                   Values are meters; -1 indicates invalid (e.g., horizon).
        u, v:      pixel coordinates in the high-res panorama image (same W x H you used).
        W, H:      width and height of the high-res panorama image.
        flipped:   True if the depth map is mirrored horizontally relative to RGB (your case).
        method:    "nearest" (fast) or "bilinear" (smoother).
        
    Returns:
        float depth in meters, or None if invalid/unavailable.
    """
    d = depth_map.data
    Hd, Wd = d.shape[:2]

    # Normalize to [0, 1]
    u01 = u / float(W)
    v01 = v / float(H)
    if not (0.0 <= u01 <= 1.0 and 0.0 <= v01 <= 1.0):
        return None  # out of bounds

    # Horizontal flip if required
    if flipped:
        u01 = 1.0 - u01

    if method == "nearest":
        ud = int(round(u01 * (Wd - 1)))
        vd = int(round(v01 * (Hd - 1)))
        val = float(d[vd, ud])
        return val if val >= 0.0 else None

    elif method == "bilinear":
        # Continuous index in low-res depth
        uf = u01 * (Wd - 1)
        vf = v01 * (Hd - 1)

        u0 = int(np.floor(uf)); u1 = min(u0 + 1, Wd - 1)
        v0 = int(np.floor(vf)); v1 = min(v0 + 1, Hd - 1)

        du = uf - u0
        dv = vf - v0

        z00 = float(d[v0, u0]); z10 = float(d[v0, u1])
        z01 = float(d[v1, u0]); z11 = float(d[v1, u1])

        # Weights for bilinear interpolation
        w00 = (1 - du) * (1 - dv)
        w10 =      du  * (1 - dv)
        w01 = (1 - du) *      dv
        w11 =      du  *      dv

        vals = np.array([z00, z10, z01, z11], dtype=float)
        wts  = np.array([w00, w10, w01, w11], dtype=float)

        # Exclude invalid samples (-1) by zeroing their weights
        wts[vals < 0.0] = 0.0
        if wts.sum() <= 0.0:
            return None

        return float((wts * vals).sum() / wts.sum())

    else:
        raise ValueError("method must be 'nearest' or 'bilinear'")

# --- WGS84 helpers: build a local ENU frame at the pano's GPS ---
_A  = 6378137.0               # semi-major axis [m]
_F  = 1 / 298.257223563
_E2 = _F * (2 - _F)           # eccentricity^2

def geodetic_to_ecef(lat, lon, h):
    s, c = math.sin(lat), math.cos(lat)
    sl, cl = math.sin(lon), math.cos(lon)
    N = _A / math.sqrt(1 - _E2 * s*s)
    X = (N + h) * c * cl
    Y = (N + h) * c * sl
    Z = (N * (1 - _E2) + h) * s
    return np.array([X, Y, Z], float)

def ecef_to_enu_matrix(lat0, lon0):
    s, c = math.sin(lat0), math.cos(lat0)
    sl, cl = math.sin(lon0), math.cos(lon0)
    E = np.array([-sl,       cl,      0.0])
    N = np.array([-s*cl, -s*sl,   c])
    U = np.array([ c*cl,  c*sl,   s])
    return np.vstack([E, N, U])

@dataclass
class LocalFrame:
    lat0: float; lon0: float; h0: float
    X0_ecef: np.ndarray; R_ecef2enu: np.ndarray

def make_local_frame(lat_deg, lon_deg, h_m=0.0):
    lat0 = math.radians(lat_deg); lon0 = math.radians(lon_deg)
    X0   = geodetic_to_ecef(lat0, lon0, h_m)
    R    = ecef_to_enu_matrix(lat0, lon0)
    return LocalFrame(lat0, lon0, h_m, X0, R)

# --- Pixel -> direction in camera, then rotate to world (ENU) ---
def Rz(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def Rx(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

def pixel_to_cam_dir(u, v, W, H):
    theta = 2*math.pi*(u/W - 0.5)        # azimuth offset [-pi,pi]
    phi   = math.pi*(0.5 - v/H)          # elevation [-pi/2,pi/2]
    d = np.array([math.cos(phi)*math.sin(theta),
                  math.sin(phi),
                  math.cos(phi)*math.cos(theta)], float)
    return d / np.linalg.norm(d)

def ray_world_dir(u, v, W, H, heading, pitch=0.0, roll=0.0):
    d_cam = pixel_to_cam_dir(u, v, W, H)
    R = Rz(heading) @ Rx(pitch) @ Rz(roll)   # yaw(heading) about Up, small pitch/roll
    d_w = R @ d_cam
    return d_w / np.linalg.norm(d_w)

# --- ENU -> lat/lon using pyproj (install: pip install pyproj) ---
def enu_to_lla(X_enu, lf: LocalFrame):
    try:
        from pyproj import CRS, Transformer
    except ImportError as e:
        raise RuntimeError("pyproj is required for ENU->LLA conversion. pip install pyproj") from e
    ecef = CRS.from_epsg(4978); lla = CRS.from_epsg(4979)
    R_enu2ecef = lf.R_ecef2enu.T
    X_ecef = lf.X0_ecef + R_enu2ecef @ X_enu
    to_lla = Transformer.from_crs(ecef, lla, always_xy=True)
    lon, lat, h = to_lla.transform(X_ecef[0], X_ecef[1], X_ecef[2])
    return lat, lon, h

# --- Main: pixel + depth -> lat/lon (and optional ground snap) ---
def localize_pixel_with_depth(pano, u, v, W, H, depth_m,
                              depth_is_slant=True,
                              snap_to_ground=False,
                              camera_height_m=2.6):
    """
    pano: object with fields lat, lon, elevation (optional), heading, pitch, roll  [radians]
    (u,v): pixel coords in the same W x H image you used
    depth_m: distance from camera (meters). From Street View depth this is a slant range.
    depth_is_slant: True if 'depth_m' is along the ray (Street View default) -> use C + s*d
    snap_to_ground: if True, drop to a flat ground plane z = ground_z near the camera
    camera_height_m: used only if snap_to_ground=True and pano.elevation is camera altitude
    """
    # 1) Local frame centered at pano GPS (use pano.elevation if present else 0)
    alt = getattr(pano, "elevation", 0.0) or 0.0
    lf  = make_local_frame(pano.lat, pano.lon, alt)

    # 2) Camera center in ENU (it's the origin of this local frame)
    C = np.zeros(3, float)

    # 3) World ray direction
    heading = getattr(pano, "heading", 0.0) or 0.0
    pitch   = getattr(pano, "pitch",   0.0) or 0.0
    roll    = getattr(pano, "roll",    0.0) or 0.0
    d = ray_world_dir(u, v, W, H, heading, pitch, roll)

    # 4) Point in ENU
    if depth_is_slant:
        X = C + depth_m * d           # slant range along the ray (Street View depth)
    else:
        # if depth is horizontal ground distance, march in horizontal azimuth only:
        d_h = d.copy(); d_h[2] = 0.0
        n = np.linalg.norm(d_h)
        if n < 1e-9:
            raise ValueError("Horizontal direction is degenerate at this pixel.")
        X = C + depth_m * (d_h / n)

    # 5) (Optional) snap to ground: assume local flat ground near the camera
    if snap_to_ground:
        ground_z = -float(camera_height_m)   # if camera z is approx ground+camera_height
        X = X.copy(); X[2] = ground_z

    # 6) Convert to lat/lon
    lat, lon, h = enu_to_lla(X, lf)
    return lat, lon
