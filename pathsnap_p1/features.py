"""Reference-free candidate/features API shared with P2/P3/backend."""
import numpy as np
import pandas as pd

from .geometry import RoadIndex,point_segment,angle_difference

FEATURES=['distance_m','heading_difference_deg','speed_mps','arc_length_m',
          'distance_rank','candidate_count','time_gap_s','previous_nearest_distance_m',
          'heading_missing','speed_missing','large_gap']
KEYS=['split','trip_id','observation_id']


def make_index(trip, config):
    arcs=trip.arcs.loc[trip.arcs.valid]
    xy=trip.nodes[['x_m','y_m']].to_numpy()
    return RoadIndex(arcs.arc_id.to_numpy(),xy[arcs.from_node.to_numpy()],
                     xy[arcs.to_node.to_numpy()],config['index_spacing_m'])


def generate_features(trip, config, index=None):
    """Does not use trip.route, reference labels, or evaluation outcomes."""
    index=make_index(trip,config) if index is None else index
    obs=trip.observations
    points=obs[['x_m','y_m']].to_numpy()
    k=max(config['candidate_top_k']+[config['final_top_k']])
    all_ids=[];all_dist=[];near_projection=[]
    for p in points:
        ids,dist=index.nearest(p,k)
        all_ids.append(ids);all_dist.append(dist)
        near_projection.append(point_segment(p,index.starts[ids[:1]],index.ends[ids[:1]])[2][0])
    counts=np.array([min(len(ids),config['final_top_k']) for ids in all_ids])
    t=np.repeat(np.arange(len(points)),counts)
    ids=np.concatenate([a[:c] for a,c in zip(all_ids,counts)])
    distances=np.concatenate([a[:c] for a,c in zip(all_dist,counts)])
    records=obs.iloc[t]
    _,fraction,projected=point_segment(points[t],index.starts[ids],index.ends[ids])
    valid_previous=(t>0)&(records.time_gap_s.to_numpy()>0)&~records.large_gap.to_numpy()
    previous=np.asarray(near_projection)[np.maximum(t-1,0)]
    previous_distance=np.where(valid_previous,np.linalg.norm(projected-previous,axis=1),np.nan)
    frame=pd.DataFrame({'split':trip.split,'trip_id':trip.trip_id,
        'observation_id':records.observation_id.to_numpy(dtype=int),'sequence_index':t,
        'timestamp':records.timestamp.to_numpy(),'candidate_arc_id':index.arc_ids[ids],
        'distance_m':distances,'heading_difference_deg':angle_difference(records.heading_deg.to_numpy(),index.headings[ids]),
        'speed_mps':records.speed_mps.to_numpy(),'arc_length_m':index.lengths[ids],
        'distance_rank':np.concatenate([np.arange(1,c+1) for c in counts]),'candidate_count':counts[t],
        'time_gap_s':records.time_gap_s.to_numpy(),'previous_nearest_distance_m':previous_distance,
        'heading_missing':records.heading_deg.isna().to_numpy(dtype=float),
        'speed_missing':records.speed_mps.isna().to_numpy(dtype=float),'large_gap':records.large_gap.to_numpy(dtype=float),
        'projected_x_m':projected[:,0],'projected_y_m':projected[:,1],
        'segment_fraction':fraction,'epsg':trip.audit['epsg']})
    return frame,all_ids,all_dist,index


def attach_labels(features, labels):
    """Labels are joined only after reference-free feature generation."""
    output=features.merge(labels,on='observation_id',validate='many_to_one')
    output['label']=np.where(output.reference_arc_id<0,-1,
                            (output.candidate_arc_id==output.reference_arc_id).astype(int))
    return output


def candidate_sweep(trip, labels, ids_by_point, distances_by_point, index, config):
    reference=labels.reference_arc_id.to_numpy()
    usable=reference>=0
    settings=[('radius',r) for r in config['candidate_radii_m']]+[('top_k',k) for k in config['candidate_top_k']]
    rows=[]
    points=trip.observations[['x_m','y_m']].to_numpy()
    for kind,value in settings:
        hit=empty=count=0
        for t,p in enumerate(points):
            ids=index.radius(p,value)[0] if kind=='radius' else ids_by_point[t][:int(value)]
            count+=len(ids);empty+=not len(ids)
            hit+=bool(usable[t] and reference[t] in index.arc_ids[ids])
        denominator=int(usable.sum())
        rows.append(dict(split=trip.split,trip_id=trip.trip_id,method=kind,value=value,
            observations=len(points),labelled_observations=denominator,
            missed=denominator-hit,miss_rate=(denominator-hit)/denominator if denominator else np.nan,
            empty_sets=empty,candidate_rows=count,mean_candidates=count/len(points)))
    return rows
