# SOLWEIG input pipeline

Scripts that build the four input rasters for SOLWEIG (UMEP) at 1 m resolution for US cities from public data:

| Raster | Contents | Source |
|---|---|---|
| `<city>_DEM.tif` | bare-earth elevation (m) | USGS 3DEP 1 m DEM |
| `<city>_DSM.tif` | ground plus buildings (m), no vegetation | DEM, with lidar roof heights inside building footprints |
| `<city>_landcover.tif` | UMEP classes: 1 paved, 2 building, 5 grass, 6 bare soil, 7 water | UrbanWatch 1 m, NOAA C-CAP 1 m, ESA WorldCover 10 m (in that order) |
| `<city>_CDSM.tif` | canopy height above ground (m) | Meta/WRI Canopy Height v2 |
| `<city>_CDSM_lidar.tif` | canopy height above ground (m) | lidar vegetation returns filtered with NAIP NDVI, Meta where there is no lidar |

Each grid is the bounding box of the city boundary plus a 500 m buffer, in the city's UTM zone (NAD83). Rasters are not clipped to the boundary. All rasters for a city share the same grid.

Written for the input data of the Cool Choices NO-HEAT VIP project.

## Setup

Python 3.11 on Windows (other platforms should work).

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Running a city

```
.venv\Scripts\python run_chain.py boulder seattle
```

`run_chain.py` runs each city in order:

1. `run_until_done.py <city>` runs `build_solweig_inputs.py` and restarts it after any crash or dropped connection. Every step resumes where it stopped.
2. `extra_buildings.py` adds Microsoft footprints that have an ML height and are missing from Overture.
3. `fix_flat_footprints.py` handles footprints with no lidar roof (see below).
4. `add_overviews.py` and `make_qgis_styles.py` add overviews and `.qml` colour styles for QGIS.
5. `make_preview.py` writes `<city>/<city>_preview.png` (full grid and a 1.4 km downtown zoom of each raster).

A finished city writes `qa/chain_<city>.done` and is skipped on the next run.

Single steps can be run directly:

```
.venv\Scripts\python build_solweig_inputs.py denver --steps lidar,dsm
.venv\Scripts\python build_solweig_inputs.py austin --estimate
```

## Steps in `build_solweig_inputs.py`

| Step | What it does |
|---|---|
| `dem` | Lists the 3DEP 1 m tiles of each project in the city config and mosaics them, first project first. Later projects fill holes. |
| `buildings` | Overture building footprints for the grid, queried with DuckDB from the Overture S3 release. |
| `lidar` | Streams lidar from USGS EPT (Entwine) or NOAA COPC. Only octree nodes that intersect the grid are read, at about 1 m point spacing. Writes three working grids: highest non-vegetation return per cell (roofs), highest vegetation-candidate return per cell, and lidar coverage. The NAIP NDVI layer is built in a parallel thread. |
| `dsm` | DEM outside footprints. Inside footprints: lidar roof elevation, then the nearest roof point of the same footprint within 30 px, then the footprint's mean roof, then the Overture height. Heights above the city cap are removed and spikes are cleaned. |
| `landcover` | UrbanWatch, then C-CAP, then WorldCover, mapped to UMEP classes. Footprints are burned in as buildings. |
| `canopy` | Meta CHM v2 tiles, set to 0 on buildings and water. |
| `canopylidar` | Lidar canopy: vegetation-candidate returns minus ground, kept only where NAIP NDVI > 0.12. Buildings (grown 2 px) and water are removed. Then: 35 m cap, despike at median + 4 m, 3x3 opening (removes wires and wall edges), pit-free closing, Gaussian smoothing (sigma 1 m) inside crowns, specks under 6 m² removed. Meta fills cells with no lidar within 15 m. |

### Working files

The lidar working grids are memory-mapped files written in random order. They are slow on an external hard drive, so when the project is not on the system drive they go to `~/solweig_work/<city>` (or `SOLWEIG_WORK` if set). They are deleted when the city finishes unless `--keep-work` is given.

## City configuration

Each city in `CITIES` (top of `build_solweig_inputs.py`) lists its boundary file, DEM projects, lidar sources and land cover sources, plus:

- `max_height`: cap on building height above ground. Returns above it are noise.
- `keep_high_noise`: keep lidar class 18. The Dallas 2019 and Pima County 2021 vendors put real tower roofs in it.
- `veg_roofs_above_m`: inside footprints, vegetation-class returns higher than this are used as roof. The 2018 Georgia statewide lidar classifies most tower roof returns as high vegetation.
- `lidar_spacing` and `detail`: coarser citywide spacing with full density around tall buildings (used for Houston).

`qa/discover_sources.py` finds the DEM projects, EPT lidar and C-CAP layers that cover a new boundary. `qa/probe_noise.py` checks how the lidar classifies points on tall towers before a full run.

### Sources by city

| City | DEM projects | Lidar | Land cover |
|---|---|---|---|
| Dallas | TX_Pecos_Dallas_2018_D19, TX_2022NorthCentral_ContributedProcessing_D25 | TX_Pecos_Dallas_B3_2018, TX_Pecos_Dallas_B2a_2018 | UrbanWatch Dallas 2016, UrbanWatch Denton County 2020, WorldCover |
| Houston | TX_Houston_B24, TX_CoastalRegion_2018_A18 | TX_Coastal_B1_2018, TX_Coastal_B2_Z15_2018, TX_Coastal_B3_2018 | C-CAP Houston 2020, WorldCover |
| Miami | FL_MiamiDade_D23, FL_Southeast_B2_2018, FL_Southeast_B1_2018 | NOAA 10338 (2021 Miami-Dade) | UrbanWatch Miami 2017, C-CAP Florida 2022, WorldCover |
| Los Angeles | CA_LosAngeles_B23, CA_LosAngeles_2016, CA_SoCal_Wildfires_B3_2018 | CA_LosAngeles_1_B23 | UrbanWatch Los Angeles 2014, C-CAP California 2021, WorldCover |
| New York City | NY_LongIsland_A24 and 7 older projects | NOAA 9306 (2017 NYC Topobathy) | UrbanWatch New York City 2017, C-CAP New York 2021, C-CAP New Jersey 2020, WorldCover |
| San Francisco | CA_SanFrancisco_B23, CA_AlamedaCounty_2021_B21, CA_CaliforniaGaps_B23, CA_NoCAL_3DEP_Supp_Funding_2018_D18 | CA_SanFrancisco_1_B23, CA_AlamedaCo_1/2/3_2021 | UrbanWatch San Francisco 2014, C-CAP California 2021, WorldCover |
| Philadelphia | DE_DelawareValley_HD_2015, DE_DelawareValleyHD_5A_2015 | NOAA 9848 (2022 Philadelphia) | UrbanWatch Philadelphia 2017, C-CAP Pennsylvania 2020, C-CAP New Jersey 2020, WorldCover |
| Tucson | AZ_PimaCounty_2021_B21, AZ_Eastern_PimaCO_2015, AZ_BrawleyRillito_2018_D19 | AZ_PimaCo_2_2021 | WorldCover |
| Boulder | CO_DRCOG_2020_B20, CO_SoPlatteRiver_Lot5_2013, CO_SoPlatteRiver_Lot2a_2013, CO_ArapahoRooseveltPikeNF_D23 | CO_DRCOG_1_2020, CO_DRCOG_2_2020 | WorldCover (UrbanWatch Denver is listed but does not reach Boulder) |
| Seattle | WA_KingCounty_2021_B21 | WA_KingCo_1_2021 | UrbanWatch Seattle 2017, C-CAP Washington 2021, WorldCover |
| Denver | CO_DRCOG_2020_B20, CO_SoPlatteRiver_Lot5_2013, CO_EasternColorado_2018_A18 | CO_DRCOG_2_2020 | UrbanWatch Denver 2017, WorldCover |
| Austin | TX_Central_B1_2017, TX_Hurricane_2018_D18_SUPPLEMENTAL_DRRA | USGS_LPC_TX_Central_B1_2017_LAS_2019 | WorldCover |
| Atlanta | GA_Statewide_2018_B18_DRRA, GA_Central_2019_B19 | GA_Statewide_B2_2018, GA_Statewide_B3_2018, GA_Central_5_2018, GA_Central_6_2018 | UrbanWatch Atlanta 2017, C-CAP Georgia 2021, WorldCover |

The Seattle boundary was supplied in Florida State Plane North (EPSG:26960) and was reprojected to UTM 10N. The original is in `qa/`.

## Footprint fixes

`fix_flat_footprints.py`: a footprint whose DSM is flat at ground level (no lidar roof) takes its height from, in order, the Overture height, the Microsoft ML height (overlap at least 0.3), or the city's typical house height. The typical height is the median lidar height of non-flat footprints of 100 to 400 m², applied to footprints of 40 to 2000 m². Anything else becomes paved. In Houston this restored 101,564 homes that had been paved.

## Checks

| Script | Check |
|---|---|
| `qa_check.py` | reads all four rasters block by block: grid alignment, nodata, value ranges |
| `make_preview.py` | full-grid and downtown preview image of a city's rasters |
| `height_benchmark.py` | DSM building heights against Wikipedia's lists of tallest buildings |
| `validation_sheets.py` | landmark sheets: photo, lidar top-down view, DSM, and a cross-section through the highest pixel |
| `datum_check.py` | vertical offset between lidar ground and the DEM |
| `canopy_check.py` | Meta canopy against lidar vegetation heights in sampled wooded squares |
| `missing_buildings.py` | UrbanWatch building areas with no Overture footprint |

Results for the first eight cities: 147 of 172 benchmark buildings within tolerance (12 of the misses were built after the lidar survey). On test blocks, canopy from 1 m lidar spacing matched full density, while 2 m spacing lost 13 to 19% of tree area and 35 to 40% of small crowns.

## Other scripts

- `extra_buildings.py`: Microsoft footprints with an ML height that Overture is missing.
- `clamp_dsm.py`: removes DSM cells above the city's height cap from an existing DSM.
- `compact_rasters.py`: rewrites rasters that were patched in place, verified block by block.
- `qa/rebuild_ndvi.py`: rebuilds one city's NDVI layer.

## Known issues

- Austin, Tucson and Boulder have no 1 m land cover product. They use WorldCover 10 m with 1 m building footprints burned in.
- Buildings built after the lidar survey take the Overture or Microsoft height, or the typical house height.
- NAIP access tokens from the Planetary Computer last about 15 minutes. `pc_token()` renews them from the token's own expiry.
- 0.3 m NAIP years are read from their 0.6 m overview. On a Boulder test block this matched full resolution with correlation 0.998.
