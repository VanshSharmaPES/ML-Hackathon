"""Development-only orchestration, audit trails, and teammate handover."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import time

import numpy as np
import pandas as pd

from .data import ALLOWED_SPLITS,discover,load_trip
from .alignment import align_reference
from .features import FEATURES,generate_features,attach_labels,candidate_sweep
from .visuals import save_trajectory_plot,save_folium_map,eda_plots


def json_write(path, value):
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,default=convert),encoding='utf-8')


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(config, limit=None):
    start=time.perf_counter()
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    inventory=[];sweep=[];all_observations=[];manifest=[];warnings=[]
    try:
        import folium
        map_available=True
    except ImportError:
        map_available=False
        warnings.append('Folium unavailable: static maps generated; interactive maps pending. Install requirements and run maps.')
    # A run manifest explicitly lists current outputs: never glob stale trips into training.
    processed=[]
    for split in ALLOWED_SPLITS:
        directories=discover(config['data_root'],split)
        if limit:directories=directories[:limit]
        for number,directory in enumerate(directories,1):
            print(f'[{split} {number}/{len(directories)}] {directory.name}',flush=True)
            trip=load_trip(directory,split,config)
            features,ids,dist,index=generate_features(trip,config)
            labels=align_reference(trip,config)
            labelled=attach_labels(features,labels)
            sweeps=candidate_sweep(trip,labels,ids,dist,index,config)
            observations=trip.observations.merge(labels,on='observation_id',validate='one_to_one')
            observations.insert(0,'trip_id',trip.trip_id);observations.insert(0,'split',split)
            present=labelled.groupby('observation_id').label.max()
            observations['reference_in_candidates']=observations.observation_id.map(present).eq(1)
            target=out/'processed'/split/trip.trip_id;target.mkdir(parents=True,exist_ok=True)
            for filename,frame in [('candidates',labelled),('observations',observations),('nodes',trip.nodes),('arcs',trip.arcs),('rejected',trip.rejected)]:
                frame.to_parquet(target/f'{filename}.parquet',index=False)
            json_write(target/'metadata.json',trip.audit)
            processed.append({'split':split,'trip_id':trip.trip_id,'directory':str(target.resolve()),'candidate_rows':len(labelled)})
            trip.audit['candidate_rows']=len(labelled)
            trip.audit['alignment_far_points']=int(labels.get('alignment_far',pd.Series(dtype=bool)).sum())
            trip.audit['alignment_ambiguous_points']=int(labels.alignment_ambiguous.sum())
            trip.audit['alignment_crossed_break_points']=int(labels.crosses_reference_break.sum())
            inventory.append(trip.audit);sweep.extend(sweeps);all_observations.append(observations)
            for p in sorted(directory.iterdir()):
                if p.suffix in ('.track','.nodes','.arcs','.route'):
                    manifest.append({'split':split,'trip_id':trip.trip_id,'file':p.name,'bytes':p.stat().st_size,'sha256':file_hash(p)})
            if number <= config['maps_per_partition']:
                save_trajectory_plot(trip,out/'figures'/f'{split}_{trip.trip_id}_trajectory.png')
                if map_available:save_folium_map(trip,out/'maps'/f'{split}_{trip.trip_id}.html')
    inventory=pd.DataFrame(inventory);sweep=pd.DataFrame(sweep)
    observations=pd.concat(all_observations,ignore_index=True)
    inventory.to_csv(out/'dataset_inventory.csv',index=False)
    sweep.to_csv(out/'candidate_sweep_by_trip.csv',index=False)
    totals=sweep.groupby(['split','method','value'],as_index=False)[['observations','labelled_observations','missed','empty_sets','candidate_rows']].sum()
    totals['miss_rate']=totals.missed/totals.labelled_observations
    totals['mean_candidates']=totals.candidate_rows/totals.observations
    totals.to_csv(out/'candidate_sweep_summary.csv',index=False)
    observations.to_parquet(out/'processed'/'observations.parquet',index=False)
    pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
    eda_plots(inventory,sweep,observations,out/'figures')
    versions={}
    for package in ['numpy','pandas','scipy','scikit-learn','matplotlib','pyarrow','joblib','pyproj','folium']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not installed'
    run={'config':config,'trips':processed,'versions':versions,'warnings':warnings,
         'elapsed_seconds':time.perf_counter()-start,'limited_run':limit is not None,
         'interactive_maps_generated':map_available,'features':FEATURES}
    json_write(out/'run_manifest.json',run)
    json_write(out/'feature_schema.json',{'features':FEATURES,'key':['split','trip_id','observation_id','candidate_arc_id'],
        'label':'1 = matches derived reference arc; 0 = other; -1 = reference unavailable',
        'projection':'Per-trip EPSG recorded in candidates and metadata; never compare x/y across trips.',
        'previous_nearest_distance_m':'Euclidean distance between previous GPS nearest-road projection and current candidate projection; missing across gaps/invalid dt.',
        'missing_policy':'Preserved as NaN; model-specific imputation learned within each training fold.',
        'selection':'Fixed top-8 road candidates by exact finite-segment distance; no ground-truth injection.',
        'alignment':'Monotonic route-occurrence DP; derived labels are imperfect and may favour distance-based methods.'})
    write_eda_report(inventory,totals,run,out)
    print(f'Prepared {len(inventory)} trips in {run["elapsed_seconds"]:.1f}s',flush=True)
    for warning in warnings:print('NOTICE:',warning,flush=True)
    return run


def write_eda_report(inventory,sweep,run,out):
    lines=['# P1 dataset exploration — measured results','',
           'These results use train and student_test only. Evaluator trips were not read. Labels are derived by monotonic reference-route alignment, not independently verified point labels.','',
           '| Partition | Trips | Raw GPS | Clean GPS | Candidate rows | Route breaks |',
           '|---|---:|---:|---:|---:|---:|']
    for split,g in inventory.groupby('split'):
        lines.append(f'| {split} | {len(g)} | {g.raw_observations.sum():,} | {g.clean_observations.sum():,} | {g.candidate_rows.sum():,} | {g.route_breaks.sum()} |')
    lines+=['','## Data quality','',
        f'- Invalid observations removed: {inventory.invalid_observations.sum():,}.',
        f'- Exact duplicate observations removed: {inventory.exact_duplicates.sum():,}.',
        f'- Duplicate timestamps retained and flagged: {inventory.duplicate_timestamps.sum():,}.',
        f'- Large gaps (> {run["config"]["gap_seconds"]:g} seconds): {inventory.large_gaps.sum():,}.',
        f'- Invalid road arcs excluded while retaining original IDs: {inventory.invalid_arcs.sum():,}.',
        f'- Reference connections that break: {inventory.route_breaks.sum():,}.',
        f'- Reference-aligned points farther than {run["config"]["alignment_warning_distance_m"]:g} m: {inventory.alignment_far_points.sum():,}.',
        f'- Geometrically ambiguous aligned points: {inventory.alignment_ambiguous_points.sum():,}.',
        f'- Used EPSG codes: {", ".join(map(str,sorted(inventory.epsg.unique())))}.',
        '', '## Projection decision','',
        'The supplied assignment suggested UTM 32N. Actual trip coordinates span multiple continents and both hemispheres. A local WGS84 UTM zone is selected per trip; GPS and nodes use the same CRS. Fixed 32N mode refuses incompatible trips. pyproj is preferred; when unavailable, a tested sixth-order WGS84 transverse-Mercator series is recorded explicitly in metadata.',
        '', '## Candidate coverage','',
        '| Partition | Search | Setting | Miss rate | Mean candidates | Empty sets |',
        '|---|---|---:|---:|---:|---:|']
    for r in sweep.itertuples():
        lines.append(f'| {r.split} | {r.method} | {r.value:g} | {r.miss_rate:.3%} | {r.mean_candidates:.2f} | {r.empty_sets} |')
    lines+=['','The reproducible baseline uses fixed top-8 candidates. This is a declared starting configuration, not a claimed optimum. Radius sweeps do not use fallback candidates; empty sets count as misses. No reference arcs are inserted.','',
        '## Alignment limitations','',
        'Dynamic programming allows repeated assignments and forward jumps through route positions. It preserves repeated arc occurrences and does not require increasing numerical arc IDs. It has no forced first/last route entry. Disconnected references, parallel roads, loops, intersections, large gaps and GPS noise can produce incorrect derived labels. Distances, ambiguity flags, and crossings of broken reference links are retained for review. A nearest-route geometric objective can favour distance-based classifiers; these scores are agreement with the derived labels, not independently verified accuracy.',
        '', '## Figures and maps','',
        'See figures/dataset_summary.png, figures/coordinates_and_intervals.png, figures/candidate_miss_rates.png, and representative trajectory figures. Interactive Folium maps contain original GPS, network, reference and gap layers, with controls and a legend. External map assets require internet.',
        '',f'Interactive Folium generation status: {"completed" if run["interactive_maps_generated"] else "pending: Folium dependency unavailable"}.',
        '', '## Feature and preprocessing decisions','',
        'See feature_schema.json and STEP_BY_STEP_GUIDE.md. Speed/heading/previous-projection features are missing across >5 s gaps, at invalid time differences, and where direction is unreliable. Exact duplicate GPS rows are removed; stationary repeated locations are retained. Road nodes/arcs are never renumbered. Median imputation and scaling are learned within each training fold.','',
        '## Reproducibility','',
        'source_manifest.csv records input hashes; run_manifest.json records settings, package versions, and every processed trip. Reported observations and plots come from the actual run.']
    (out/'EDA_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def regenerate_maps(config):
    manifest=Path(config['output_dir'])/'run_manifest.json'
    if not manifest.exists():raise FileNotFoundError('Run prepare first.')
    run=json.loads(manifest.read_text(encoding='utf-8'))
    import folium
    counts={s:0 for s in ALLOWED_SPLITS}
    for item in run['trips']:
        split=item['split']
        if counts[split] >= config['maps_per_partition']:continue
        trip=load_trip(Path(config['data_root'])/split/item['trip_id'],split,config)
        save_folium_map(trip,Path(config['output_dir'])/'maps'/f'{split}_{trip.trip_id}.html')
        counts[split]+=1
    run['interactive_maps_generated']=True
    run['warnings']=[w for w in run['warnings'] if not w.startswith('Folium unavailable')]
    run['versions']['folium']=importlib.metadata.version('folium')
    json_write(manifest,run)
    write_eda_report(pd.read_csv(Path(config['output_dir'])/'dataset_inventory.csv'),
                     pd.read_csv(Path(config['output_dir'])/'candidate_sweep_summary.csv'),run,Path(config['output_dir']))
    print('Generated Folium maps:',counts,flush=True)
