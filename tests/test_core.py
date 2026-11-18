from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB

from pathsnap_p1.geometry import RoadIndex,point_segment,project_utm,choose_utm,angle_difference
from pathsnap_p1.alignment import monotonic_alignment
from pathsnap_p1.data import load_trip,discover
from pathsnap_p1.features import generate_features,FEATURES
from pathsnap_p1.models import evaluate_predictions,make_pipeline

CONFIG=json.loads((Path(__file__).parents[1]/'config.json').read_text())


def test_projection_against_published_proj_examples():
    # https://proj.org/en/stable/operations/projections/utm.html (GRS80/WGS84 difference negligible here)
    x,_=project_utm([12],[56],32632)
    np.testing.assert_allclose(x,[[687071.44,6210141.33]],atol=.03,rtol=0)
    x,_=project_utm([174],[-44],32759)
    np.testing.assert_allclose(x,[[740526.32,5123750.87]],atol=.03,rtol=0)
    assert choose_utm([8.5],[49])==32632
    assert choose_utm([151],[-34])==32756
    with pytest.raises(ValueError):project_utm([151],[-34],32632)


def test_kdtree_matches_bruteforce_and_finds_long_segments():
    rng=np.random.default_rng(17)
    a=rng.uniform(-500,500,(200,2));b=rng.uniform(-500,500,(200,2))
    a[0]=[-5000,0];b[0]=[5000,0]
    index=RoadIndex(np.arange(len(a)),a,b,50)
    for p in [np.array([0,.1])]+list(rng.uniform(-1000,1000,(15,2))):
        d,_,_=point_segment(p,a,b)
        expected=np.lexsort((np.arange(len(a)),d))[:8]
        ids,dist=index.nearest(p,8)
        np.testing.assert_array_equal(ids,expected)
        np.testing.assert_allclose(dist,d[expected])
        ids,_=index.radius(p,100)
        assert set(ids)==set(np.flatnonzero(d<=100))


def test_finite_segment_and_heading_wrap():
    d,f,p=point_segment(np.array([3,4]),np.array([[0,0]]),np.array([[1,0]]))
    assert d[0]==pytest.approx(np.sqrt(20))
    assert f[0]==1
    assert angle_difference(359,1)==2


def test_alignment_matches_exhaustive_optimum():
    from itertools import combinations_with_replacement
    starts=np.array([[0,0],[10,0],[20,0]],float);ends=starts+[8,0]
    points=np.array([[1,1],[12,1],[7,1],[25,1]],float)
    chosen=monotonic_alignment(points,starts,ends)
    costs=np.array([point_segment(p,starts,ends)[0]**2 for p in points])
    best=min(sum(costs[t,j] for t,j in enumerate(seq)) for seq in combinations_with_replacement(range(3),4))
    assert sum(costs[t,j] for t,j in enumerate(chosen))==pytest.approx(best)
    assert np.all(np.diff(chosen)>=0)


def sample_trip(tmp_path):
    d=tmp_path/'trip';d.mkdir()
    (d/'trip.track').write_text('9 50 1\n9 50 1\n9 50 2\n9.001 50 20\n')
    (d/'trip.nodes').write_text('9 50\n9.001 50\n9.002 50\n')
    (d/'trip.arcs').write_text('0 1\n1 2\n1 0\n')
    (d/'trip.route').write_text('0\n1\n')
    return d


def test_cleaning_preserves_stationary_points_and_ids(tmp_path):
    trip=load_trip(sample_trip(tmp_path),'train',CONFIG)
    assert trip.observations.observation_id.tolist()==[0,2,3]
    assert trip.audit['exact_duplicates']==1
    assert trip.observations.iloc[-1].large_gap
    assert np.isnan(trip.observations.iloc[-1].speed_mps)
    assert trip.arcs.arc_id.tolist()==[0,1,2]


def test_reference_cannot_affect_features(tmp_path):
    directory=sample_trip(tmp_path)
    original=load_trip(directory,'train',CONFIG,with_reference=False)
    a=generate_features(original,CONFIG)[0]
    (directory/'trip.route').write_text('NOT A VALID ROUTE')
    other=load_trip(directory,'train',CONFIG,with_reference=False)
    b=generate_features(other,CONFIG)[0]
    pd.testing.assert_frame_equal(a,b)
    assert not {'label','reference_arc_id','route_position'} & set(FEATURES)


def test_candidate_miss_counts_as_top1_failure():
    frame=pd.DataFrame({'split':['student_test']*2,'trip_id':['a','b'],
         'observation_id':[0,0],'candidate_arc_id':[2,2],'distance_rank':[1,1],'label':[1,0]})
    obs=pd.DataFrame({'split':['student_test']*3,'trip_id':['a','b','c'],
         'observation_id':[0,0,0],'reference_arc_id':[2,3,4],'reference_in_candidates':[True,False,False]})
    m,_,_=evaluate_predictions(frame,[.9,.2],obs)
    assert m['top1_accuracy']==pytest.approx(1/3)
    assert m['candidate_miss_rate']==pytest.approx(2/3)


def test_scaling_is_fit_on_training_only_and_groups_do_not_overlap():
    X=pd.DataFrame({'x':[0.,2.,100.,102.]});y=np.array([0,1,0,1]);groups=np.array(['a','a','b','b'])
    for tr,va in GroupKFold(2).split(X,y,groups):
        assert not set(groups[tr])&set(groups[va])
        p=make_pipeline(GaussianNB()).fit(X.iloc[tr],y[tr])
        assert p.named_steps['scaler'].mean_[0]==pytest.approx(X.iloc[tr].x.mean())


def test_evaluator_partition_is_rejected(tmp_path):
    with pytest.raises(ValueError):discover(tmp_path,'evaluator_test')
    with pytest.raises(ValueError):load_trip(tmp_path,'evaluator_test',CONFIG)


def test_invalid_network_entries_preserve_ids(tmp_path):
    directory=sample_trip(tmp_path)
    (directory/'trip.arcs').write_text('0 1\ninf 1\n1 2\n')
    (directory/'trip.route').write_text('0\n2\n')
    trip=load_trip(directory,'train',CONFIG)
    assert trip.audit['invalid_arcs']==1
    assert trip.arcs.loc[trip.arcs.valid,'arc_id'].tolist()==[0,2]


def test_repeated_route_occurrences_are_not_collapsed():
    starts=np.array([[0.,0.],[10.,0.],[0.,0.]])
    ends=np.array([[8.,0.],[18.,0.],[8.,0.]])
    points=np.array([[1.,0.],[12.,0.],[1.,0.]])
    assert monotonic_alignment(points,starts,ends).tolist()==[0,1,2]
