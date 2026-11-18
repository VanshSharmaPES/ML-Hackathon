"""Reproducible inspection of highest candidate-miss trips (no model changes)."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from pathsnap_p1.data import load_trip
from pathsnap_p1.geometry import point_segment
from pathsnap_p1.visuals import save_trajectory_plot,save_folium_map


def main():
    base=Path(__file__).parent
    config=json.loads((base/'config.json').read_text())
    root=(base/config['data_root']).resolve();out=base/config['output_dir']
    sweep=pd.read_csv(out/'candidate_sweep_by_trip.csv',dtype={'trip_id':str})
    worst=sweep.loc[(sweep.method=='top_k')&(sweep.value==config['final_top_k'])].nlargest(2,'missed')
    results=[]
    for row in worst.itertuples():
        trip=load_trip(root/row.split/row.trip_id,row.split,config)
        save_trajectory_plot(trip,out/'figures'/f'{row.split}_{row.trip_id}_anomaly.png')
        save_folium_map(trip,out/'maps'/f'{row.split}_{row.trip_id}_anomaly.html')
        labels=pd.read_parquet(out/'processed'/row.split/row.trip_id/'observations.parquet')
        arcs=trip.arcs.iloc[trip.route]
        nodes=trip.nodes[['x_m','y_m']].to_numpy()
        starts=nodes[arcs.from_node.to_numpy()];ends=nodes[arcs.to_node.to_numpy()]
        nearest=[]
        for p in trip.observations[['x_m','y_m']].to_numpy():
            nearest.append(point_segment(p,starts,ends)[0].min())
        results.append({'split':row.split,'trip_id':row.trip_id,'observations':row.observations,
            'candidate_misses':row.missed,'miss_rate':row.miss_rate,
            'median_nearest_reference_distance_m':np.median(nearest),
            'median_monotonic_reference_distance_m':labels.alignment_distance_m.median(),
            'nearest_reference_over_50m':int((np.array(nearest)>50).sum()),
            'monotonic_reference_over_50m':int((labels.alignment_distance_m>50).sum())})
        print('Reviewed',row.trip_id,flush=True)
    result=pd.DataFrame(results);result.to_csv(out/'alignment_anomalies.csv',index=False)
    lines=['# Alignment anomaly review','',
        'These trips were selected by the largest number of top-8 candidate misses on permitted development data. No observations were dropped and no labels were manually changed.','',
        '| Trip | Points | Candidate misses | Median nearest-reference distance | Median ordered-alignment distance |',
        '|---|---:|---:|---:|---:|']
    for r in result.itertuples():
        lines.append(f'| {r.trip_id} | {r.observations:,} | {r.candidate_misses:,} | {r.median_nearest_reference_distance_m:.2f} m | {r.median_monotonic_reference_distance_m:.2f} m |')
    lines+=['','Interpretation: a high nearest-reference distance indicates geometric disagreement even before order constraints. A substantially higher monotonic distance indicates that reference order also creates difficulty. Neither diagnostic establishes whether the GPS, reference route or alignment assumptions are wrong; inspect the maps and discuss with organisers.','',
        'This explains why a model can have good student-test coverage but much weaker performance on a particular training fold. Retain the documented limitations when reporting model metrics.']
    (out/'ALIGNMENT_ANOMALIES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
