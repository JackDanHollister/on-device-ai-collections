#!/usr/bin/env python3
"""
Coordinate Utilities
====================

Handles grid reference detection and conversion to WGS84 coordinates.
Supports UK National Grid (BNG) and other coordinate systems.
"""

import re
import math
from typing import Dict, Optional, List, Tuple

# Ordnance Survey, A Guide to Coordinate Systems in Great Britain v3.6:
# sections 6.6, 7.1 and appendices B/C. OSTN15 is needed for survey accuracy.
BNG_TRANSFORM_SOURCE = (
    "https://www.ordnancesurvey.co.uk/documents/resources/"
    "guide-coordinate-systems-great-britain.pdf"
)
BNG_TRANSFORM_METHOD = "osgb36_to_wgs84_7_parameter_helmert"
# OS quotes 3.5 m (95%) for the transformation within Britain. Use a 5 m
# allowance alongside grid precision, not as a guaranteed error bound or a
# claim about the accuracy of the original specimen label.
BNG_TRANSFORM_ALLOWANCE_M = 5.0

# Generate the complete 100 km lettering grid, which omits I. The previous
# hand-written subset had incorrect NB, NG and OV origins and missed Shetland.
_GRID_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
GRID_SQUARES = {}
for _first, _first_letter in enumerate(_GRID_LETTERS):
    for _second, _second_letter in enumerate(_GRID_LETTERS):
        _east = ((_first - 2) % 5) * 5 + _second % 5
        _north = 19 - (_first // 5) * 5 - _second // 5
        if 0 <= _east < 7 and 0 <= _north < 13:
            GRID_SQUARES[_first_letter + _second_letter] = (
                _east * 100000, _north * 100000
            )

_GRID_BODY = (
    r"(?P<square>[A-HJ-Z]{2})\s*/?\s*"
    r"(?:(?P<east>[0-9]{2,5})[\s/]+(?P<north>[0-9]{2,5})"
    r"|(?P<digits>[0-9]{4,10}))"
)
_GRID_REFERENCE_RE = re.compile(_GRID_BODY, re.IGNORECASE)
_GRID_DETECT_RE = re.compile(
    r"(?<!\w)" + _GRID_BODY + r"(?!\w)(?![\s/]+[0-9])", re.IGNORECASE
)


def bng_to_wgs84(easting: float, northing: float) -> Tuple[float, float]:
    """
    Convert British National Grid coordinates to WGS84 lat/long.
    Inverse Airy 1830 Transverse Mercator projection followed by an approximate
    seven-parameter Helmert transformation, assuming zero source height.
    OS quotes 3.5 m (95%) datum-transformation accuracy within Britain; this is
    not OSTN15 and is not suitable for survey accuracy or offshore positioning.
    
    Args:
        easting: BNG easting in meters
        northing: BNG northing in meters
        
    Returns:
        Tuple of (latitude, longitude) in WGS84 decimal degrees

    Raises:
        ValueError: nonfinite or out-of-grid input. The grid rectangle is a
            numeric domain check, not a guarantee that a point is on land.
    """
    if isinstance(easting, bool) or isinstance(northing, bool):
        raise ValueError("BNG coordinates must be finite numbers")
    try:
        easting, northing = float(easting), float(northing)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("BNG coordinates must be finite numbers") from exc
    if not (
        math.isfinite(easting) and math.isfinite(northing)
        and 0 <= easting <= 700000 and 0 <= northing <= 1300000
    ):
        raise ValueError("Coordinates lie outside the British National Grid")
    lat_osgb, lon_osgb = _bng_to_osgb36(easting, northing)
    return _osgb36_to_wgs84(lat_osgb, lon_osgb)


def _bng_to_osgb36(easting: float, northing: float) -> Tuple[float, float]:
    """Inverse National Grid projection; return Airy 1830 angles in radians."""
    # OSGB36 ellipsoid parameters
    a = 6377563.396      # Semi-major axis
    b = 6356256.909      # Semi-minor axis
    f0 = 0.9996012717    # Scale factor
    lat0 = math.radians(49.0)  # True origin latitude
    lon0 = math.radians(-2.0)  # True origin longitude
    n0 = -100000         # Northing of true origin
    e0 = 400000          # Easting of true origin
    
    e2 = 1 - (b*b)/(a*a)
    n = (a-b)/(a+b)
    
    # Iterative calculation for latitude
    lat = lat0
    M = 0
    
    for _ in range(20):
        lat = ((northing - n0 - M) / (a * f0)) + lat
        Ma = (1 + n + (5/4)*n*n + (5/4)*n*n*n) * (lat - lat0)
        Mb = (3*n + 3*n*n + (21/8)*n*n*n) * math.sin(lat - lat0) * math.cos(lat + lat0)
        Mc = ((15/8)*n*n + (15/8)*n*n*n) * math.sin(2*(lat - lat0)) * math.cos(2*(lat + lat0))
        Md = (35/24)*n*n*n * math.sin(3*(lat - lat0)) * math.cos(3*(lat + lat0))
        M = b * f0 * (Ma - Mb + Mc - Md)
        if abs(northing - n0 - M) < 0.00001:
            break
    else:
        raise ValueError("National Grid inverse projection did not converge")
    
    # Calculate longitude
    cosLat = math.cos(lat)
    sinLat = math.sin(lat)
    nu = a * f0 / math.sqrt(1 - e2 * sinLat * sinLat)
    rho = a * f0 * (1 - e2) / ((1 - e2 * sinLat * sinLat) ** 1.5)
    eta2 = nu/rho - 1
    
    tanLat = math.tan(lat)
    tan2lat = tanLat * tanLat
    tan4lat = tan2lat * tan2lat
    tan6lat = tan4lat * tan2lat
    
    secLat = 1.0/cosLat
    nu3 = nu*nu*nu
    nu5 = nu3*nu*nu
    nu7 = nu5*nu*nu
    
    VII = tanLat / (2 * rho * nu)
    VIII = tanLat / (24 * rho * nu3) * (5 + 3*tan2lat + eta2 - 9*tan2lat*eta2)
    IX = tanLat / (720 * rho * nu5) * (61 + 90*tan2lat + 45*tan4lat)
    
    X = secLat / nu
    XI = secLat / (6 * nu3) * (nu/rho + 2*tan2lat)
    XII = secLat / (120 * nu5) * (5 + 28*tan2lat + 24*tan4lat)
    XIIA = secLat / (5040 * nu7) * (61 + 662*tan2lat + 1320*tan4lat + 720*tan6lat)
    
    dE = easting - e0
    dE2 = dE * dE
    dE3 = dE2 * dE
    dE4 = dE2 * dE2
    dE5 = dE4 * dE
    dE6 = dE4 * dE2
    dE7 = dE6 * dE
    
    lat_osgb = lat - VII*dE2 + VIII*dE4 - IX*dE6
    lon_osgb = lon0 + X*dE - XI*dE3 + XII*dE5 - XIIA*dE7
    
    return lat_osgb, lon_osgb


def _osgb36_to_wgs84(
    latitude: float, longitude: float, height: float = 0.0
) -> Tuple[float, float]:
    """Airy radians to WGS84 degrees using OS's approximate inverse Helmert."""
    a, b = 6377563.396, 6356256.909
    e2 = 1.0 - b * b / (a * a)
    nu = a / math.sqrt(1.0 - e2 * math.sin(latitude) ** 2)
    x = (nu + height) * math.cos(latitude) * math.cos(longitude)
    y = (nu + height) * math.cos(latitude) * math.sin(longitude)
    z = ((1.0 - e2) * nu + height) * math.sin(latitude)

    # Position-vector convention; inverse of OS table 4. Reversing the signs
    # is a first-order inverse, consistent with the small-angle Helmert model.
    tx, ty, tz = 446.448, -125.157, 542.060
    scale = 1.0 - 20.4894e-6
    rx, ry, rz = (math.radians(value / 3600.0) for value in (0.1502, 0.2470, 0.8421))
    x, y, z = (
        tx + scale * x - rz * y + ry * z,
        ty + rz * x + scale * y - rx * z,
        tz - ry * x + rx * y + scale * z,
    )
    a = 6378137.0
    flattening = 1.0 / 298.257223563
    e2 = flattening * (2.0 - flattening)
    p = math.hypot(x, y)
    latitude = math.atan2(z, p * (1.0 - e2))
    for _ in range(20):
        nu = a / math.sqrt(1.0 - e2 * math.sin(latitude) ** 2)
        updated = math.atan2(z + e2 * nu * math.sin(latitude), p)
        if abs(updated - latitude) < 1e-12:
            return math.degrees(updated), math.degrees(math.atan2(y, x))
        latitude = updated
    raise ValueError("WGS84 latitude conversion did not converge")


def _parse_grid_reference(grid_ref: str) -> Optional[Tuple[float, float, float]]:
    if not isinstance(grid_ref, str):
        return None
    text = re.sub(
        r"^(?:grid\s*ref(?:erence)?|GR)\s*[:=]?\s*", "", grid_ref.strip(),
        flags=re.IGNORECASE,
    )
    match = _GRID_REFERENCE_RE.fullmatch(text)
    if match is None or match.group("square").upper() not in GRID_SQUARES:
        return None
    if match.group("digits") is not None:
        digits = match.group("digits")
        if len(digits) % 2:
            return None
        east, north = digits[:len(digits) // 2], digits[len(digits) // 2:]
    else:
        east, north = match.group("east"), match.group("north")
        if len(east) != len(north):
            return None
    resolution = 10 ** (5 - len(east))
    origin_e, origin_n = GRID_SQUARES[match.group("square").upper()]
    return (
        origin_e + (int(east) + 0.5) * resolution,
        origin_n + (int(north) + 0.5) * resolution,
        float(resolution),
    )


def convert_grid_reference(grid_ref: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Convert UK National Grid reference to WGS84 coordinates.
    
    Args:
        grid_ref: Grid reference string (e.g., 'SK4480', 'TL534672', 'TL 199 798')
        
    Returns:
        Tuple of (latitude, longitude, radius_meters) or (None, None, None) if conversion fails
        The coordinates represent the CENTER of the grid square. The radius is
        its half-diagonal plus a 5 m datum-transformation allowance, not total
        specimen-position uncertainty or a guaranteed confidence interval.
    """
    parsed = _parse_grid_reference(grid_ref)
    if parsed is None:
        return None, None, None
    easting, northing, resolution = parsed
    precision_radius = resolution / math.sqrt(2) + BNG_TRANSFORM_ALLOWANCE_M
    lat, lon = bng_to_wgs84(easting, northing)
    return lat, lon, precision_radius


def detect_grid_references(locality: str) -> List[str]:
    """
    Detect supported British National Grid references in stable text order.
    Irish and other grids are deliberately not interpreted as BNG.
    
    Args:
        locality: Raw locality string
        
    Returns:
        List of detected grid references
    """
    if not isinstance(locality, str):
        return []
    found_grids, seen = [], set()
    for match in _GRID_DETECT_RE.finditer(locality):
        reference = match.group(0)
        parsed = _parse_grid_reference(reference)
        if parsed is not None and parsed not in seen:
            found_grids.append(reference)
            seen.add(parsed)
    return found_grids


def validate_coordinates(lat: str, lon: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Validate and convert latitude/longitude strings to floats.
    
    Args:
        lat: Latitude string
        lon: Longitude string
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if invalid
    """
    try:
        if lat is None or lon is None or isinstance(lat, bool) or isinstance(lon, bool):
            return None, None
            
        lat_val = float(str(lat).strip())
        lon_val = float(str(lon).strip())
        
        # Check if coordinates are within valid ranges
        if math.isfinite(lat_val) and math.isfinite(lon_val) and -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
            return lat_val, lon_val
        else:
            return None, None
            
    except (ValueError, TypeError):
        return None, None


def preprocess_coordinates(record: Dict) -> Dict:
    """
    Analyze a record and determine the best coordinate source.
    Priority: existing coordinates > grid conversion > needs AI processing
    
    Args:
        record: Museum record dictionary
        
    Returns:
        Enhanced record with preprocessing results
    """
    from .field_mapping import get_existing_coordinates, get_locality

    locality = get_locality(record)
    existing_lat, existing_lon = get_existing_coordinates(record)
    
    # Check for existing valid coordinates first
    lat, lon = validate_coordinates(existing_lat, existing_lon)
    if lat is not None and lon is not None:
        record['preprocessed_source'] = 'coordinates_provided'
        record['preprocessed_lat'] = lat
        record['preprocessed_lon'] = lon
        record['preprocessed_confidence'] = 'high'
        return record
    
    # Look for grid references in locality text
    detected_grids = detect_grid_references(locality)
    if len(detected_grids) > 1:
        record['preprocessed_coordinate_issue'] = 'MULTIPLE_GRID_REFERENCES'
        detected_grids = []
    
    for grid_ref in detected_grids:
        lat, lon, radius = convert_grid_reference(grid_ref)
        if lat is not None and lon is not None:
            record['preprocessed_source'] = 'grid_reference_converted'
            record['preprocessed_lat'] = lat
            record['preprocessed_lon'] = lon
            record['preprocessed_radius'] = radius
            record['preprocessed_confidence'] = 'high'
            record['converted_grid_ref'] = grid_ref
            record['coordinate_transformation'] = BNG_TRANSFORM_METHOD
            record['coordinate_transformation_allowance_m'] = BNG_TRANSFORM_ALLOWANCE_M
            return record
    
    # No coordinates found - will need AI processing
    record['preprocessed_source'] = 'needs_ai_processing'
    record['preprocessed_confidence'] = 'unknown'
    return record


if __name__ == "__main__":
    # Test the coordinate utilities
    print("Testing Coordinate Utilities")
    print("=" * 40)
    
    # Test grid reference conversion
    test_grids = ["SK4480", "TL199798", "SU306053", "INVALID123"]
    
    for grid in test_grids:
        lat, lon, radius = convert_grid_reference(grid)
        if lat and lon:
            print(f"SUCCESS {grid} -> {lat:.6f}, {lon:.6f} (+/-{radius:.1f}m)")
        else:
            print(f"ERROR {grid} -> Failed to convert")
    
    # Test detection
    test_locality = "Monks Wood, Hunts., England, TL199798, elevation 50m"
    detected = detect_grid_references(test_locality)
    print(f"\nDetected grids in '{test_locality}': {detected}")
    
    # Test record preprocessing
    test_record = {
        'Barcode': '10577259',
        'Locality verbatim': 'Monks Wood, Hunts., England, TL199798',
        'Country': 'United Kingdom'
    }
    
    enhanced = preprocess_coordinates(test_record)
    print(f"\nRecord preprocessing:")
    print(f"   Source: {enhanced.get('preprocessed_source')}")
    print(f"   Coordinates: {enhanced.get('preprocessed_lat')}, {enhanced.get('preprocessed_lon')}")
    print(f"   Confidence: {enhanced.get('preprocessed_confidence')}")
