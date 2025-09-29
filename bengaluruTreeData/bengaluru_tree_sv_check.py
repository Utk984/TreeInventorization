# bengaluru_tree_sv_check.py
# Requires: pip install streetlevel pandas tqdm lxml
# Example (one line):
#   python bengaluru_tree_sv_check.py --kml data/bbmp_tree_census_july2025.kml --out bengaluru_tree_sv_results.csv --radius 60 --pause 1.0 --batch-size 500 --search-third-party 0

import math
import time
import csv
import os
import re
import argparse
import warnings
from typing import Dict, Iterator, Tuple

import pandas as pd
from tqdm import tqdm
from lxml import etree, html as lxml_html
from streetlevel import streetview


# ------------------ Utilities ------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    from math import radians, sin, cos, asin, sqrt
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = phi2 - phi1
    dlmb = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlmb/2)**2
    return 2 * R * asin(sqrt(a))


def _norm_key(s: str) -> str:
    """lowercase + strip + remove non-alphanum; e.g., 'Common Name' -> 'commonname'."""
    return re.sub(r'[^a-z0-9]+', '', (s or '').strip().lower())


# Broad synonym sets for field names seen in KML/CSV exports
NAME_KEYS = {
    'treename', 'name', 'commonname', 'localname', 'vernacularname', 'kannadaname',
    'speciescommonname', 'treelabel', 'tree_no', 'treeno', 'treecode'
}
# include many species/scientific variants
SPECIES_KEYS = {
    'species', 'scientificname', 'botanicalname', 'speciesname', 'genus', 'genusspecies',
    'sciname', 'scientific', 'sname', 'species_scientific', 'species(scientific)',
    'scientificnames', 'botanicalnames', 'scientific_name', 'botanical_name'
}

# common author abbreviations hinting a scientific (Latin) string
_LATIN_AUTHORS = (' L.', ' (L.)', ' Roxb', ' Benth', ' Del.', ' Linn', ' Gaertn', ' Willd', ' DC.', ' R.Br')


def _looks_scientific(s: str) -> bool:
    """Heuristic: detect scientific/Latin names like 'Ficus religiosa L.' or 'Albizia Lebbeck (L.) Benth.'"""
    if not s or len(s) < 4:
        return False
    if any(tok in s for tok in _LATIN_AUTHORS):
        return True
    # Capitalized genus + at least one more token (species/author parts)
    return bool(re.match(r'^[A-Z][a-z]+(?:\s+[A-Za-z\.\-\(\)]+){1,}$', s.strip()))


def extract_tree_props(elem: etree._Element) -> Dict:
    """
    Pull likely tree properties (name/species) from a KML Placemark.
    - Reads ExtendedData/Data and SchemaData/SimpleData
    - Uses Placemark <name>
    - Parses <description> HTML (tables and "Key: Value" lines)
    - Normalizes keys and searches synonym sets
    - If species missing but tree_name looks scientific, species := tree_name
    """
    raw = {}

    # 1) ExtendedData/Data(name/value)
    for data_elem in elem.findall(".//{http://www.opengis.net/kml/2.2}Data"):
        name = data_elem.get("name")
        val = data_elem.findtext("{http://www.opengis.net/kml/2.2}value")
        if name and val:
            raw[name] = val

    # 2) SchemaData/SimpleData
    for sd in elem.findall(".//{http://www.opengis.net/kml/2.2}SimpleData"):
        name = sd.get("name")
        if name and sd.text:
            raw[name] = sd.text

    # 3) Placemark name
    pm_name = elem.findtext("{http://www.opengis.net/kml/2.2}name")
    if pm_name:
        raw.setdefault("name", pm_name)

    # 4) Description HTML (tables or "Key: Value")
    desc = elem.findtext("{http://www.opengis.net/kml/2.2}description")
    if desc:
        try:
            doc = lxml_html.fromstring(desc)
            # Table rows: first cell -> key, second -> value
            for tr in doc.xpath(".//tr"):
                tds = tr.xpath(".//td")
                if len(tds) >= 2:
                    key = ''.join(tds[0].itertext()).strip()
                    val = ''.join(tds[1].itertext()).strip()
                    if key and val:
                        raw.setdefault(key, val)
            # Paragraphs "Key: Value"
            for p in doc.xpath(".//p"):
                txt = ' '.join(p.itertext()).strip()
                m = re.match(r'\s*([^:]+)\s*:\s*(.+)$', txt)
                if m:
                    raw.setdefault(m.group(1), m.group(2))
        except Exception:
            # description may be plain text or malformed HTML — ignore safely
            pass

    # Normalize keys
    norm = {_norm_key(k): v for k, v in raw.items() if isinstance(v, str) and v.strip()}

    # Choose best candidates
    species = None
    for k in SPECIES_KEYS:
        if k in norm:
            species = norm[k]
            break

    tree_name = None
    for k in NAME_KEYS:
        if k in norm:
            tree_name = norm[k]
            break
    if not tree_name and 'name' in norm:
        tree_name = norm['name']  # fallback

    # Fallback: if species missing but tree_name looks scientific, use it
    if not species and tree_name and _looks_scientific(tree_name):
        species = tree_name

    return {"tree_name": tree_name, "species": species}


def stream_kml_points(path: str) -> Iterator[Tuple[float, float, Dict]]:
    """
    Stream Placemark points from KML without loading entire file.
    Yields: (lat, lon, props_dict)
    """
    ns = "{http://www.opengis.net/kml/2.2}"
    context = etree.iterparse(path, events=("end",), tag=f"{ns}Placemark")
    for _, pm in context:
        try:
            point = pm.find(f".//{ns}Point")
            if point is None:
                pm.clear()
                continue
            coords_text = point.findtext(f"{ns}coordinates")
            if not coords_text:
                pm.clear()
                continue
            lon, lat, *_ = map(float, coords_text.split(","))
            props = extract_tree_props(pm)
            yield lat, lon, props
        except Exception as e:
            warnings.warn(f"Skipping a Placemark due to parse error: {e}")
        finally:
            # Free memory progressively
            pm.clear()
            while pm.getprevious() is not None:
                del pm.getparent()[0]


def nearest_streetview(
    lat: float,
    lon: float,
    radius_m: int = 60,
    search_third_party: bool = False,
    max_retries: int = 4,
    base_sleep: float = 1.0,
):
    """
    Query Street View with retries + backoff.
    Returns (has_sv, dist_m, pano_dict)
    """
    attempt = 0
    while True:
        try:
            pano = streetview.find_panorama(
                lat=lat, lon=lon, radius=radius_m, search_third_party=search_third_party
            )
            if pano is None:
                return False, None, None
            dist = haversine_m(lat, lon, pano.lat, pano.lon)
            pano_dict = {
                "pano_id": pano.id,
                "pano_lat": pano.lat,
                "pano_lon": pano.lon,
                "pano_date": getattr(pano, "date", None),
                "street_name": getattr(pano, "street_name", None),
                "is_third_party": getattr(pano, "is_third_party", None),
                "permalink": pano.permalink() if hasattr(pano, "permalink") else None,
            }
            return True, float(dist), pano_dict
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                warnings.warn(f"Street View lookup failed at {lat},{lon} after retries: {e}")
                return False, None, None
            time.sleep(base_sleep * (2 ** (attempt - 1)))


def load_done_keys(csv_path: str) -> set:
    """For resume: read existing CSV and collect processed lat/lon keys."""
    if not os.path.exists(csv_path):
        return set()
    done = set()
    for chunk in pd.read_csv(csv_path, usecols=["tree_lat", "tree_lon"], chunksize=100_000):
        for lat, lon in zip(chunk["tree_lat"], chunk["tree_lon"]):
            done.add((round(float(lat), 7), round(float(lon), 7)))
    return done


def write_batch(csv_path: str, rows: list, header_written: bool):
    """Append a batch to CSV; write header if new."""
    if not rows:
        return header_written
    fieldnames = [
        "tree_name", "species", "tree_lat", "tree_lon",
        "has_streetview", "sv_distance_m",
        "pano_id", "pano_lat", "pano_lon", "pano_date",
        "street_name", "is_third_party", "permalink",
    ]
    mode = "a" if os.path.exists(csv_path) and header_written else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
            header_written = True
        writer.writerows(rows)
    return header_written


# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kml", default="/mnt/data/bbmp_tree_census_july2025.kml")
    ap.add_argument("--out", default="bengaluru_tree_sv_results.csv")
    ap.add_argument("--radius", type=int, default=60)
    ap.add_argument("--pause", type=float, default=1.0, help="Pause between requests (s)")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--search-third-party", type=int, default=0, help="1 to include third-party panoramas")
    args = ap.parse_args()

    search_third_party = bool(args.search_third_party)

    # Resume support
    done_keys = load_done_keys(args.out)
    header_written = os.path.exists(args.out)

    batch = []
    total_processed = 0
    total_skipped = 0

    try:
        for lat, lon, props in tqdm(stream_kml_points(args.kml), desc="Scanning trees"):
            key = (round(lat, 7), round(lon, 7))
            if key in done_keys:
                total_skipped += 1
                continue

            try:
                has_sv, dist_m, pano = nearest_streetview(
                    lat, lon, radius_m=args.radius, search_third_party=search_third_party
                )
            except Exception as e:
                warnings.warn(f"Lookup error at {lat},{lon}: {e}")
                has_sv, dist_m, pano = False, None, None

            row = {
                "tree_name": props.get("tree_name"),
                "species": props.get("species"),
                "tree_lat": lat,
                "tree_lon": lon,
                "has_streetview": bool(has_sv),
                "sv_distance_m": round(dist_m, 2) if dist_m is not None else None,
                "pano_id": pano.get("pano_id") if pano else None,
                "pano_lat": pano.get("pano_lat") if pano else None,
                "pano_lon": pano.get("pano_lon") if pano else None,
                "pano_date": str(pano.get("pano_date")) if pano else None,
                "street_name": pano.get("street_name") if pano else None,
                "is_third_party": pano.get("is_third_party") if pano else None,
                "permalink": pano.get("permalink") if pano else None,
            }
            batch.append(row)
            done_keys.add(key)
            total_processed += 1

            # Flush periodically
            if len(batch) >= args.batch_size:
                header_written = write_batch(args.out, batch, header_written)
                batch.clear()

            time.sleep(args.pause)

    except KeyboardInterrupt:
        warnings.warn("Interrupted by user. Flushing partial results...")

    finally:
        # Final flush
        header_written = write_batch(args.out, batch, header_written)
        print(
            f"Done. Wrote/updated: {args.out}\n"
            f"Processed new: {total_processed:,} | Skipped existing: {total_skipped:,}"
        )


if __name__ == "__main__":
    main()
