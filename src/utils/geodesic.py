import math
import numpy as np
from dataclasses import dataclass

# ------------------ Depth lookup (robust) ------------------

def get_depth_at_pixel(depth_map, u, v, W, H, flipped=True, method="bilinear"):
    """
    Return depth (meters) at pano pixel (u,v) from a low-res depth map.
    - Values < 0 are invalid and ignored.
    - 'flipped=True' mirrors horizontally BEFORE interpolation (recommended).
    """
    d = depth_map.data
    Hd, Wd = d.shape[:2]

    # Normalize to [0,1]
    u01 = u / float(W)
    v01 = v / float(H)
    if not (0.0 <= u01 <= 1.0 and 0.0 <= v01 <= 1.0):
        return None

    # Apply the horizontal flip on normalized coord first
    if flipped:
        u01 = 1.0 - u01

    if method == "nearest":
        ud = int(round(u01 * (Wd - 1)))
        vd = int(round(v01 * (Hd - 1)))
        val = float(d[vd, ud])
        return val if val >= 0.0 else None

    # Bilinear sampling (recommended)
    uf = u01 * (Wd - 1)
    vf = v01 * (Hd - 1)
    u0 = int(math.floor(uf)); v0 = int(math.floor(vf))
    u1 = min(u0 + 1, Wd - 1); v1 = min(v0 + 1, Hd - 1)
    du = uf - u0; dv = vf - v0

    z00 = float(d[v0, u0]); z10 = float(d[v0, u1])
    z01 = float(d[v1, u0]); z11 = float(d[v1, u1])

    w00 = (1 - du) * (1 - dv)
    w10 =      du  * (1 - dv)
    w01 = (1 - du) *      dv
    w11 =      du  *      dv

    vals = np.array([z00, z10, z01, z11], dtype=float)
    wts  = np.array([w00, w10, w01, w11], dtype=float)

    # Ignore invalid neighbors
    wts[vals < 0.0] = 0.0
    s = wts.sum()
    if s <= 0.0:
        return None
    return float((wts * vals).sum() / s)

# ------------------ Local ENU utilities ------------------

_A  = 6378137.0               # WGS84 semi-major axis [m]
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
    E = np.array([-sl,      cl,    0.0])
    N = np.array([-s*cl, -s*sl,    c ])
    U = np.array([ c*cl,  c*sl,    s ])
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

# ------------------ Camera ray construction ------------------

def pixel_to_cam_dir(u, v, W, H):
    """
    Equirectangular pixel -> CAMERA-FRAME unit direction.
    Camera axes: x=right, y=up, z=forward.
    """
    theta = 2*math.pi*(u/W - 0.5)       # azimuth offset [-pi,pi]
    phi   = math.pi*(0.5 - v/H)         # elevation [-pi/2,pi/2] (+up)
    d = np.array([math.cos(phi)*math.sin(theta),
                  math.sin(phi),
                  math.cos(phi)*math.cos(theta)], float)
    return d / np.linalg.norm(d)

def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

def ray_world_dir(u, v, W, H, heading, pitch=0.0, roll=0.0):
    """
    1) pixel -> camera-frame dir
    2) world-from-camera rotation: yaw(heading) about Up, then pitch about camera X, then roll about camera Z
    """
    d_cam = pixel_to_cam_dir(u, v, W, H)
    R_world_from_cam = Rz(heading) @ Rx(pitch) @ Rz(roll)
    d_world = R_world_from_cam @ d_cam
    return d_world / np.linalg.norm(d_world)

# ------------------ Ground-constrained localization ------------------

def intersect_ray_with_horizontal_plane(C, d, z_plane):
    """Return intersection of ray X=C+t d with plane z=z_plane (ENU)."""
    if abs(d[2]) < 1e-9:
        return None  # parallel to plane
    t = (z_plane - C[2]) / d[2]
    if t <= 0:
        return None  # behind camera
    return C + t*d

def localize_ground_pixel(pano, u, v, W, H,
                          lf: LocalFrame,
                          depth_map=None, flipped=True,
                          default_camera_height=2.6,
                          use_depth_to_estimate_height=True,
                          plausible_height_range=(1.2, 4.5)):
    """
    Localize a GROUND point: intersect the viewing ray with a ground plane z = -h_cam.
    If depth_map is provided, estimate h_cam ≈ -s * d_z (robust) at this pixel; else use default.
    Returns: (lat, lon)
    """
    # Camera center at ENU origin for this pano's local frame
    C = np.zeros(3, float)

    heading = getattr(pano, "heading", 0.0) or 0.0
    pitch   = getattr(pano, "pitch",   0.0) or 0.0
    roll    = getattr(pano, "roll",    0.0) or 0.0
    d = ray_world_dir(u, v, W, H, heading, pitch, roll)

    # Estimate camera height from depth at this ground pixel (optional)
    h_cam = float(default_camera_height)
    if depth_map is not None and use_depth_to_estimate_height:
        s = get_depth_at_pixel(depth_map, u, v, W, H, flipped=flipped, method="bilinear")
        if s is not None and abs(d[2]) > 1e-6:
            h_est = - s * d[2]  # since ground_z = s*d_z, and ground_z = -h_cam (camera at z=0)
            if plausible_height_range[0] <= h_est <= plausible_height_range[1]:
                h_cam = float(h_est)

    Xg = intersect_ray_with_horizontal_plane(C, d, z_plane=-h_cam)
    if Xg is None:
        raise ValueError("Ray does not hit ground plane in front of camera (pixel too close to horizon).")

    lat, lon, _ = enu_to_lla(Xg, lf)
    return lat, lon

# ------------------ (Optional) slant-depth localization for non-ground targets ------------------

def localize_pixel_with_slant_depth(pano, u, v, W, H, depth_m, lf: LocalFrame):
    """
    Use slant depth along the ray (non-ground targets).
    """
    C = np.zeros(3, float)
    heading = getattr(pano, "heading", 0.0) or 0.0
    pitch   = getattr(pano, "pitch",   0.0) or 0.0
    roll    = getattr(pano, "roll",    0.0) or 0.0
    d = ray_world_dir(u, v, W, H, heading, pitch, roll)
    X = C + float(depth_m) * d
    lat, lon, _ = enu_to_lla(X, lf)
    return lat, lon
