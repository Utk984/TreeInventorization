import math
import logging
import time
import numpy as np
from dataclasses import dataclass

# Configure logger for geodesic calculations
logger = logging.getLogger(__name__)

def get_depth_at_pixel(depth_map, u, v, W, H, flipped=True):
    """
    Simple depth lookup from depth map at pixel coordinates.
    
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

    # Simple coordinate mapping: scale high-res coords to depth map coords
    ud = int(u * Wd / W)
    vd = int(v * Hd / H)
    
    # Bounds check
    if ud < 0 or ud >= Wd or vd < 0 or vd >= Hd:
        return None

    # Horizontal flip if required
    if flipped:
        ud = Wd - 1 - ud

    val = float(d[vd, ud])
    return val if val >= 0.0 else None

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
