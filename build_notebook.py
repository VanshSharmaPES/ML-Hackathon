"""Create the reproducible review notebook; results remain in audited output files."""
from pathlib import Path
import json


def main():
    cells=[]
    def markdown(text):
        cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                      'source':text.splitlines(keepends=True)})
    markdown('# PathSnap P1 — dataset and feature review\n\nRun `python -m pathsnap_p1 prepare` first. This notebook reads actual generated results; it does not fit preprocessing or access evaluator data. Read STEP_BY_STEP_GUIDE.md and HANDOVER.md alongside it.')
    code("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, IFrame\nroot = Path.cwd()\nif root.name == 'notebooks':\n    root = root.parent\nout = root / 'outputs'\ninventory = pd.read_csv(out / 'dataset_inventory.csv', dtype={'trip_id': str})\nsweep = pd.read_csv(out / 'candidate_sweep_summary.csv')\ndisplay(inventory.groupby('split').agg(trips=('trip_id','size'), gps=('raw_observations','sum'), candidate_rows=('candidate_rows','sum')))\n")
    markdown('## Raw file meanings and cleaning\n\n`.track`: longitude, latitude, timestamp; `.nodes`: longitude, latitude; `.arcs`: node connections; `.route`: ordered reference arc IDs. Original row IDs are retained. Exact duplicate GPS records are removed; repeated locations at different times remain.')
    code("display(inventory)\ndisplay(inventory[['split','trip_id','invalid_observations','exact_duplicates','duplicate_timestamps','large_gaps','route_breaks']].sort_values('large_gaps', ascending=False).head(15))\ndisplay(Image(filename=str(out / 'figures/dataset_summary.png')))\ndisplay(Image(filename=str(out / 'figures/coordinates_and_intervals.png')))\n")
    markdown('## Candidate coverage\n\nAll distances are metres in a suitable per-trip UTM CRS. Radius sets may be empty. Top-k is exact finite-segment distance. Reference candidates are never inserted. Scores use derived monotonic-alignment labels.')
    code("display(sweep)\ndisplay(Image(filename=str(out / 'figures/candidate_miss_rates.png')))\n")
    markdown('## Trace one observation\n\nRows share a GPS observation and differ by candidate road. Keep target/quality metadata out of the model feature matrix.')
    code("first = inventory.loc[inventory.split.eq('train')].iloc[0]\ntrip_dir = out / 'processed' / 'train' / first.trip_id\ncandidates = pd.read_parquet(trip_dir / 'candidates.parquet')\nobservations = pd.read_parquet(trip_dir / 'observations.parquet')\ndisplay(observations.head())\ndisplay(candidates.loc[candidates.observation_id.eq(observations.observation_id.iloc[0])])\ndisplay(json.loads((out / 'feature_schema.json').read_text()))\n")
    markdown('## Geographical review\n\nThe interactive maps require Folium at generation time and internet for external browser assets. Static figures use real projected geometry. Reference layers are only for development data.')
    code("display(Image(filename=str(out / 'figures/train_00000001_trajectory.png')))\nmap_path = out / 'maps/train_00000001.html'\nif map_path.exists():\n    display(IFrame(src='../outputs/maps/train_00000001.html', width='100%', height=550))\nelse:\n    print('Run python -m pathsnap_p1 maps after installing Folium.')\n")
    markdown('## Actual model comparisons\n\nRun `python -m pathsnap_p1 train` first. Candidate macro F1 is binary; observation Top-1 includes candidate misses. Training imputation/scaling are fitted inside each trip-grouped CV fold.')
    code("comparison = out / 'model_comparison.csv'\nif comparison.exists():\n    display(pd.read_csv(comparison))\n    display(pd.read_csv(out / 'cv_results.csv'))\n    display(Image(filename=str(out / 'figures/model_comparison.png')))\nelse:\n    print('Model experiments have not completed yet.')\n")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python','version':'3.14'}},'nbformat':4,'nbformat_minor':5}
    path=Path(__file__).parent/'notebooks'/'01_data_exploration.ipynb'
    path.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    print(path)


if __name__=='__main__':main()
