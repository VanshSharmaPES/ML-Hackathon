"""Derived labels only. This module must not be called by prediction code."""
import numpy as np
import pandas as pd

from .geometry import point_segment


def monotonic_alignment(points, starts, ends):
    """Minimise sum of squared point/segment distances with nondecreasing route positions.

    D[t,j] = distance(t,j)^2 + min(D[t-1,k] for k <= j).
    No forced start/end and no assumption that numerical arc IDs are ordered.
    Repeated arc occurrences stay separate. Gaps can skip route entries.
    """
    n, m = len(points),len(starts)
    if not n or not m:
        raise ValueError('Alignment requires observations and route entries.')
    back = np.empty((n,m),dtype=np.int32)
    previous = np.zeros(m)
    for t,point in enumerate(points):
        distances,_,_=point_segment(point,starts,ends)
        prefix = np.minimum.accumulate(previous)
        new_best = np.r_[True, previous[1:] < prefix[:-1]]
        argmin = np.maximum.accumulate(np.where(new_best,np.arange(m),0))
        back[t]=argmin
        previous=distances**2+prefix
    position=np.empty(n,dtype=int)
    position[-1]=int(np.argmin(previous))
    for t in range(n-1,0,-1):
        position[t-1]=back[t,position[t]]
    return position


def align_reference(trip, config):
    obs=trip.observations
    result=pd.DataFrame({'observation_id':obs.observation_id,'reference_arc_id':-1,
                         'route_position':-1,'alignment_distance_m':np.nan,
                         'alignment_ambiguous':False,'crosses_reference_break':False})
    if trip.route is None:
        return result
    route_arcs=trip.arcs.iloc[trip.route]
    xy=trip.nodes[['x_m','y_m']].to_numpy()
    starts=xy[route_arcs.from_node.to_numpy()]
    ends=xy[route_arcs.to_node.to_numpy()]
    points=obs[['x_m','y_m']].to_numpy()
    position=monotonic_alignment(points,starts,ends)
    assigned_dist=np.empty(len(points)); ambiguous=np.zeros(len(points),dtype=bool)
    for t,p in enumerate(points):
        d,_,_=point_segment(p,starts,ends)
        assigned_dist[t]=d[position[t]]
        # Different arc IDs within 2 m of the selected cost indicate geometric ambiguity.
        ambiguous[t]=np.any((trip.route != trip.route[position[t]]) & (d <= assigned_dist[t]+2))
    broken=np.r_[0,(route_arcs.to_node.to_numpy()[:-1] != route_arcs.from_node.to_numpy()[1:]).astype(int)]
    cumulative=np.cumsum(broken)
    crossed=np.r_[False,np.diff(cumulative[position])>0]
    result['reference_arc_id']=trip.route[position]
    result['route_position']=position
    result['alignment_distance_m']=assigned_dist
    result['alignment_ambiguous']=ambiguous
    result['crosses_reference_break']=crossed
    result['alignment_far']=assigned_dist > config['alignment_warning_distance_m']
    return result
