# P1 dataset exploration — measured results

These results use train and student_test only. Evaluator trips were not read. Labels are derived by monotonic reference-route alignment, not independently verified point labels.

| Partition | Trips | Raw GPS | Clean GPS | Candidate rows | Route breaks |
|---|---:|---:|---:|---:|---:|
| student_test | 10 | 19,160 | 19,160 | 153,280 | 1 |
| train | 80 | 208,417 | 208,417 | 1,667,336 | 13 |

## Data quality

- Invalid observations removed: 0.
- Exact duplicate observations removed: 0.
- Duplicate timestamps retained and flagged: 0.
- Large gaps (> 5 seconds): 1,260.
- Invalid road arcs excluded while retaining original IDs: 0.
- Reference connections that break: 14.
- Reference-aligned points farther than 50 m: 23,298.
- Geometrically ambiguous aligned points: 53,870.
- Used EPSG codes: 32614, 32615, 32616, 32617, 32622, 32629, 32630, 32631, 32632, 32633, 32634, 32635, 32636, 32637, 32638, 32639, 32640, 32641, 32643, 32648, 32651, 32653, 32721, 32722, 32735.

## Projection decision

The supplied assignment suggested UTM 32N. Actual trip coordinates span multiple continents and both hemispheres. A local WGS84 UTM zone is selected per trip; GPS and nodes use the same CRS. Fixed 32N mode refuses incompatible trips. pyproj is preferred; when unavailable, a tested sixth-order WGS84 transverse-Mercator series is recorded explicitly in metadata.

## Candidate coverage

| Partition | Search | Setting | Miss rate | Mean candidates | Empty sets |
|---|---|---:|---:|---:|---:|
| student_test | radius | 25 | 1.482% | 5.10 | 225 |
| student_test | radius | 50 | 0.141% | 10.91 | 0 |
| student_test | radius | 100 | 0.089% | 27.74 | 0 |
| student_test | top_k | 3 | 4.916% | 3.00 | 0 |
| student_test | top_k | 5 | 1.733% | 5.00 | 0 |
| student_test | top_k | 8 | 0.344% | 8.00 | 0 |
| train | radius | 25 | 13.215% | 4.44 | 21389 |
| train | radius | 50 | 11.166% | 8.96 | 18230 |
| train | radius | 100 | 9.854% | 22.36 | 1413 |
| train | top_k | 3 | 14.953% | 3.00 | 0 |
| train | top_k | 5 | 12.250% | 5.00 | 0 |
| train | top_k | 8 | 9.072% | 8.00 | 0 |

The reproducible baseline uses fixed top-8 candidates. This is a declared starting configuration, not a claimed optimum. Radius sweeps do not use fallback candidates; empty sets count as misses. No reference arcs are inserted.

## Alignment limitations

Dynamic programming allows repeated assignments and forward jumps through route positions. It preserves repeated arc occurrences and does not require increasing numerical arc IDs. It has no forced first/last route entry. Disconnected references, parallel roads, loops, intersections, large gaps and GPS noise can produce incorrect derived labels. Distances, ambiguity flags, and crossings of broken reference links are retained for review. A nearest-route geometric objective can favour distance-based classifiers; these scores are agreement with the derived labels, not independently verified accuracy.

## Figures and maps

See figures/dataset_summary.png, figures/coordinates_and_intervals.png, figures/candidate_miss_rates.png, and representative trajectory figures. Interactive Folium maps contain original GPS, network, reference and gap layers, with controls and a legend. External map assets require internet.

Interactive Folium generation status: completed.

## Feature and preprocessing decisions

See feature_schema.json and STEP_BY_STEP_GUIDE.md. Speed/heading/previous-projection features are missing across >5 s gaps, at invalid time differences, and where direction is unreliable. Exact duplicate GPS rows are removed; stationary repeated locations are retained. Road nodes/arcs are never renumbered. Median imputation and scaling are learned within each training fold.

## Reproducibility

source_manifest.csv records input hashes; run_manifest.json records settings, package versions, and every processed trip. Reported observations and plots come from the actual run.
