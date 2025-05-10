import rasterio
import rasterio.features
import rasterio.warp
from rasterio.transform import rowcol
from enum import Enum


class LandUseCategory(Enum):
    """Land use categories from the global land cover dataset."""
    BROADLEAF_EVERGREEN_FOREST = 1
    BROADLEAF_DECIDUOUS_FOREST = 2
    NEEDLELEAF_EVERGREEN_FOREST = 3
    NEEDLELEAF_DECIDUOUS_FOREST = 4
    MIXED_FOREST = 5
    TREE_OPEN = 6
    SHRUB = 7
    HERBACEOUS = 8
    HERBACEOUS_WITH_SPARSE_TREE_SHRUB = 9
    SPARSE_VEGETATION = 10
    CROPLAND = 11
    PADDY_FIELD = 12
    CROPLAND_OTHER_VEGETATION_MOSAIC = 13
    MANGROVE = 14
    WETLAND = 15
    BARE_AREA_CONSOLIDATED = 16
    BARE_AREA_UNCONSOLIDATED = 17
    URBAN = 18
    SNOW_ICE = 19
    WATER_BODIES = 20

    @classmethod
    def get_description(cls, value: int) -> str:
        """Get the description for a land use category value."""
        descriptions = {
            1: "Broadleaf Evergreen Forest",
            2: "Broadleaf Deciduous Forest",
            3: "Needleleaf Evergreen Forest",
            4: "Needleleaf Deciduous Forest",
            5: "Mixed Forest",
            6: "Tree Open",
            7: "Shrub",
            8: "Herbaceous",
            9: "Herbaceous with Sparse Tree/Shrub",
            10: "Sparse vegetation",
            11: "Cropland",
            12: "Paddy field",
            13: "Cropland / Other Vegetation Mosaic",
            14: "Mangrove",
            15: "Wetland",
            16: "Bare area,consolidated(gravel,rock)",
            17: "Bare area,unconsolidated (sand)",
            18: "Urban",
            19: "Snow / Ice",
            20: "Water bodies"
        }
        return descriptions.get(value, "Unknown")

def get_raster_value_at_coords(lat: float, lon: float, raster_path: str) -> float:
    """
    Get the raster value at specific latitude and longitude coordinates.
    
    Args:
        lat (float): Latitude in decimal degrees
        lon (float): Longitude in decimal degrees
        raster_path (str): Path to the GeoTIFF file
    
    Returns:
        float: The raster value at the specified coordinates
    """
    with rasterio.open(raster_path) as dataset:
        # Convert lat/lon to row/col
        row, col = rowcol(dataset.transform, lon, lat)
        
        # Read the value at the specified coordinates
        try:
            value = dataset.read(1, window=((row, row+1), (col, col+1)))[0][0]
        except IndexError:
            print(f"IndexError at {lat}, {lon}")
            value = 0
        
        return value


def is_water(lat: float, lon: float) -> bool:
    """Check if location is on water using land use raster."""
    land_use = get_raster_value_at_coords(
        lat, lon, 'data/gm_lc_v3_1_1.tif'
    )
    return land_use == LandUseCategory.WATER_BODIES.value


def is_forest(lat: float, lon: float) -> bool:
    """Check if location is in forest using land use raster."""
    land_use = get_raster_value_at_coords(
        lat, lon, 'data/gm_lc_v3_1_1.tif'
    )
    return land_use in [
        LandUseCategory.BROADLEAF_EVERGREEN_FOREST.value,
        LandUseCategory.BROADLEAF_DECIDUOUS_FOREST.value,
        LandUseCategory.NEEDLELEAF_EVERGREEN_FOREST.value,
        LandUseCategory.NEEDLELEAF_DECIDUOUS_FOREST.value,
        LandUseCategory.MIXED_FOREST.value
    ]


def get_altitude(lat: float, lon: float) -> float:
    """Get altitude from AW3D30 raster."""
    return get_raster_value_at_coords(lat, lon, 'data/output_AW3D30.tif')