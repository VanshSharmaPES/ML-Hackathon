"""Compare all prepared node coordinates with independently installed PROJ."""
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer


def main():
    out=Path(__file__).parent/'outputs'
    inventory=pd.read_csv(out/'dataset_inventory.csv',dtype={'trip_id':str})
    rows=[]
    for r in inventory.itertuples():
        nodes=pd.read_parquet(out/'processed'/r.split/r.trip_id/'nodes.parquet')
        x,y=Transformer.from_crs(4326,r.epsg,always_xy=True).transform(nodes.longitude,nodes.latitude)
        error=np.hypot(x-nodes.x_m,y-nodes.y_m)
        rows.append({'split':r.split,'trip_id':r.trip_id,'epsg':r.epsg,'nodes':len(nodes),
                     'max_error_m':float(error.max()),'mean_error_m':float(error.mean())})
    results=pd.DataFrame(rows);results.to_csv(out/'projection_verification.csv',index=False)
    maximum=results.max_error_m.max()
    print(f'Checked {results.nodes.sum():,} road nodes across {len(results)} trips; max difference from PROJ: {maximum:.6f} m')
    if maximum>.05:raise AssertionError('Projection difference exceeds 5 cm review threshold')


if __name__=='__main__':main()
