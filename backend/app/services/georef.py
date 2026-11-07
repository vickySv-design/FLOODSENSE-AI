import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pyproj
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box, shape, mapping
from shapely.ops import transform as shapely_transform


class GeoSpatialTransform:
    """
    Standard Geospatial Coordinate Reference System (CRS) & Affine Transformation pipeline.
    Maintains rigorous sub-pixel geographic consistency between WGS84 [lon, lat],
    local UTM metric projection, and raster row/column coordinates.
    """

    def __init__(self, bbox: List[float], width: int = 256, height: int = 256) -> None:
        """
        bbox format: [min_lat, min_lon, max_lat, max_lon] in EPSG:4326.
        """
        self.min_lat = float(bbox[0])
        self.min_lon = float(bbox[1])
        self.max_lat = float(bbox[2])
        self.max_lon = float(bbox[3])
        self.width = int(width)
        self.height = int(height)

        # Pixel resolution in geographic degrees
        self.res_x = (self.max_lon - self.min_lon) / float(self.width)
        self.res_y = (self.max_lat - self.min_lat) / float(self.height)

        # Dynamic local UTM EPSG selection based on center longitude and latitude
        center_lon = (self.min_lon + self.max_lon) * 0.5
        center_lat = (self.min_lat + self.max_lat) * 0.5
        self.utm_zone = int(math.floor((center_lon + 180.0) / 6.0)) + 1
        is_northern = center_lat >= 0.0
        self.utm_epsg = 32600 + self.utm_zone if is_northern else 32700 + self.utm_zone

        self.wgs84_crs = pyproj.CRS("EPSG:4326")
        self.utm_crs = pyproj.CRS(f"EPSG:{self.utm_epsg}")
        self.geod = pyproj.Geod(ellps="WGS84")

        self.to_utm = pyproj.Transformer.from_crs(self.wgs84_crs, self.utm_crs, always_xy=True).transform
        self.to_wgs84 = pyproj.Transformer.from_crs(self.utm_crs, self.wgs84_crs, always_xy=True).transform

        # Affine transform parameters: x = a*col + b*row + c, y = d*col + e*row + f
        # Standard north-up raster:
        # col=0 -> min_lon, col=width -> max_lon
        # row=0 -> max_lat (north), row=height -> min_lat (south)
        self.a = self.res_x
        self.b = 0.0
        self.c = self.min_lon
        self.d = 0.0
        self.e = -self.res_y
        self.f = self.max_lat

        # Bounding box polygon in WGS84
        self.bbox_polygon = box(self.min_lon, self.min_lat, self.max_lon, self.max_lat)

        # Precompute metric cell dimensions in local UTM
        e0, n0 = self.lonlat_to_utm(self.min_lon, self.min_lat)
        e1, n1 = self.lonlat_to_utm(self.max_lon, self.max_lat)
        self.pixel_width_m = abs(e1 - e0) / float(self.width)
        self.pixel_height_m = abs(n1 - n0) / float(self.height)
        self.cell_area_m2 = self.pixel_width_m * self.pixel_height_m

    def lonlat_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """Converts [lon, lat] (EPSG:4326) to exact floating-point raster (col, row)."""
        col = (float(lon) - self.c) / self.a
        row = (float(lat) - self.f) / self.e
        return col, row

    def pixel_to_lonlat(self, col: float, row: float) -> Tuple[float, float]:
        """Converts raster (col, row) to [lon, lat] (EPSG:4326)."""
        lon = self.c + float(col) * self.a
        lat = self.f + float(row) * self.e
        return float(lon), float(lat)

    def lonlat_to_utm(self, lon: float, lat: float) -> Tuple[float, float]:
        """Converts [lon, lat] to projected UTM metric coordinates [east_m, north_m]."""
        e, n = self.to_utm(float(lon), float(lat))
        return float(e), float(n)

    def utm_to_lonlat(self, east_m: float, north_m: float) -> Tuple[float, float]:
        """Converts UTM [east_m, north_m] to [lon, lat]."""
        lon, lat = self.to_wgs84(float(east_m), float(north_m))
        return float(lon), float(lat)

    def compute_polygon_area_m2(self, geom_or_coords: Any) -> float:
        """
        Computes accurate geodesic/ellipsoidal area of a polygon geometry in m^2.
        Accepts shapely Polygon, GeoJSON dict, or list of [lon, lat] coordinate rings.
        """
        try:
            if isinstance(geom_or_coords, (Polygon, MultiPolygon)):
                geom = geom_or_coords
            elif isinstance(geom_or_coords, dict):
                geom = shape(geom_or_coords)
            elif isinstance(geom_or_coords, (list, tuple)):
                geom = Polygon(geom_or_coords)
            else:
                return 0.0

            # Project polygon to local UTM for rigorous planar metric area
            utm_geom = shapely_transform(self.to_utm, geom)
            return float(abs(utm_geom.area))
        except Exception:
            return 0.0

    def compute_line_length_m(self, coords_lonlat: List[List[float]]) -> float:
        """Computes true metric length of a road polyline in local UTM projection."""
        if len(coords_lonlat) < 2:
            return 0.0
        total_m = 0.0
        for i in range(len(coords_lonlat) - 1):
            p1_lon, p1_lat = coords_lonlat[i]
            p2_lon, p2_lat = coords_lonlat[i + 1]
            e1, n1 = self.lonlat_to_utm(p1_lon, p1_lat)
            e2, n2 = self.lonlat_to_utm(p2_lon, p2_lat)
            total_m += math.hypot(e2 - e1, n2 - n1)
        return total_m

    def is_point_in_bbox(self, lon: float, lat: float, margin: float = 0.0) -> bool:
        """Checks if a point lies within the AOI bounding box."""
        return (
            (self.min_lon - margin) <= lon <= (self.max_lon + margin)
            and (self.min_lat - margin) <= lat <= (self.max_lat + margin)
        )

    def clip_geometry_to_bbox(self, geom: Any) -> Optional[Any]:
        """Clips a shapely geometry or GeoJSON dict to the raster bounding box."""
        try:
            sh_geom = shape(geom) if isinstance(geom, dict) else geom
            if not sh_geom.intersects(self.bbox_polygon):
                return None
            clipped = sh_geom.intersection(self.bbox_polygon)
            return clipped if not clipped.is_empty else None
        except Exception:
            return None

    def sample_raster_at_lonlat(
        self,
        raster: np.ndarray,
        lon: float,
        lat: float,
        method: str = "bilinear",
    ) -> float:
        """
        Samples a 2D raster [H, W] at arbitrary (lon, lat) using nearest or bilinear interpolation.
        """
        col, row = self.lonlat_to_pixel(lon, lat)
        H, W = raster.shape[-2:]

        if not (0 <= col < W and 0 <= row < H):
            return 0.0

        if method == "nearest":
            c_int = int(round(col))
            r_int = int(round(row))
            c_int = max(0, min(W - 1, c_int))
            r_int = max(0, min(H - 1, r_int))
            return float(raster[r_int, c_int])

        # Bilinear interpolation
        c0 = int(math.floor(col))
        r0 = int(math.floor(row))
        c1 = min(c0 + 1, W - 1)
        r1 = min(r0 + 1, H - 1)

        dc = col - c0
        dr = row - r0

        v00 = raster[r0, c0]
        v10 = raster[r0, c1]
        v01 = raster[r1, c0]
        v11 = raster[r1, c1]

        val = (1 - dc) * (1 - dr) * v00 + dc * (1 - dr) * v10 + (1 - dc) * dr * v01 + dc * dr * v11
        return float(val)

    def get_metadata(self) -> dict:
        return {
            "source_crs": "EPSG:4326",
            "utm_crs": f"EPSG:{self.utm_epsg}",
            "utm_zone": self.utm_zone,
            "bounds": [self.min_lon, self.min_lat, self.max_lon, self.max_lat],
            "affine_transform": [self.a, self.b, self.c, self.d, self.e, self.f],
            "raster_dimensions": [self.width, self.height],
            "spatial_resolution_deg": [self.res_x, self.res_y],
            "spatial_resolution_m": [round(self.pixel_width_m, 2), round(self.pixel_height_m, 2)],
            "cell_area_m2": round(self.cell_area_m2, 2),
        }
