"""Validate actual generated handover data, without fitting models again."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from pathsnap_p1.features import FEATURES


def main():
    root=Path(__file__).parent/'outputs'
    manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))
    counts={'train':0,'student_test':0};total=0
    assert not manifest['limited_run'], 'Full verification must use all supplied trips'
    for item in manifest['trips']:
        split,tid=item['split'],item['trip_id']
        assert split in counts
        counts[split]+=1
        directory=root/'processed'/split/tid
        c=pd.read_parquet(directory/'candidates.parquet')
        o=pd.read_parquet(directory/'observations.parquet')
        a=pd.read_parquet(directory/'arcs.parquet')
        assert not c.duplicated(['observation_id','candidate_arc_id']).any(),tid
        assert set(c.observation_id)==set(o.observation_id),tid
        assert c.groupby('observation_id').size().eq(8).all(),tid
        assert c.candidate_arc_id.isin(a.loc[a.valid,'arc_id']).all(),tid
        assert c.groupby('observation_id').label.sum().le(1).all(),tid
        assert c.loc[c.label==1,'candidate_arc_id'].equals(c.loc[c.label==1,'reference_arc_id']),tid
        assert not np.isinf(c[FEATURES].to_numpy()).any(),tid
        assert np.all(np.diff(o.route_position)>=0),tid
        assert c.segment_fraction.between(0,1).all(),tid
        assert c.epsg.nunique()==1,tid
        total+=len(c)
    assert counts=={'train':80,'student_test':10},counts
    print(json.dumps({'trip_counts':counts,'candidate_rows':total,'status':'passed'},indent=2))


if __name__=='__main__':main()
