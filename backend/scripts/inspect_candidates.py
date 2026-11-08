import glob, json, os, rasterio

print('--- CHECKING RASTER CANDIDATES ---')
rasters = [
    'backend/data/processed/sentinel1_preprocessed.tif',
    'backend/data/processed/sentinel2_preprocessed.tif',
    'backend/data/processed/valid_data_mask.tif',
    'backend/outputs/flood_event_analysis/potential_inundation_mask.tif',
    'backend/outputs/bolivia_holdout/masks/Bolivia_103757_pred_mask.tif',
    'backend/outputs/bolivia_holdout/masks/Bolivia_129334_pred_mask.tif',
    'backend/outputs/sen1floods11_inference/masks/India_399883_pred_mask.tif'
] + glob.glob('backend/data/satellite_cache/flood_mask_*.tif')

for p in rasters:
    if os.path.exists(p):
        with rasterio.open(p) as src:
            print(f"{p}:\n  bounds={src.bounds}\n  crs={src.crs}\n  shape={src.shape}\n  nodata={src.nodata}\n  res={src.res}\n")

print('\n--- CHECKING PRESET METADATA ---')
for p in glob.glob('backend/data/presets/**/metadata.json', recursive=True):
    with open(p, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        print(f"{p}:\n  id={meta.get('id')}\n  name={meta.get('name')}\n  bbox={meta.get('bbox')}\n")

print('\n--- CHECKING OSM CACHE FILES ---')
for p in sorted(glob.glob('backend/data/osm_cache/*.geojson')):
    sz = os.path.getsize(p)
    with open(p, 'r', encoding='utf-8') as f:
        try:
            d = json.load(f)
            fc = len(d.get('features', []))
            bbox = d.get('metadata', {}).get('bbox')
            print(f"{p}: size={sz:,} bytes, features={fc}, bbox={bbox}")
        except Exception as e:
            print(f"{p}: size={sz:,} bytes (error: {e})")

