import argparse
import json
from pathlib import Path

from .pipeline import prepare,regenerate_maps


def main():
    parser=argparse.ArgumentParser(description='PathSnap P1 data/features and baseline experiments')
    parser.add_argument('command',choices=['prepare','train','maps','all'])
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--data-root',help='Directory containing train/ and student_test/')
    parser.add_argument('--output',help='Output directory (relative to working directory)')
    parser.add_argument('--limit',type=int,help='Smoke run: first N trips per split (separate output required)')
    parser.add_argument('--folds',type=int,help='Override number of group CV folds')
    parser.add_argument('--resume',action='store_true',help='Reuse completed CV folds with matching data/config signature')
    args=parser.parse_args()
    config_path=Path(args.config).resolve()
    config=json.loads(config_path.read_text(encoding='utf-8'))
    base=config_path.parent
    config['data_root']=str(Path(args.data_root).resolve() if args.data_root else (base/config['data_root']).resolve())
    config['output_dir']=str(Path(args.output).resolve() if args.output else (base/config['output_dir']).resolve())
    if args.limit is not None and (args.limit<1 or not args.output):
        parser.error('--limit requires a positive number and a separate --output directory')
    if args.folds is not None:config['cv_folds']=args.folds
    if config['cv_folds']<2:parser.error('At least two CV folds are required')
    if args.command in ['prepare','all']:prepare(config,args.limit)
    if args.command in ['train','all']:
        from .models import train
        train(config,resume=args.resume)
    if args.command=='maps':regenerate_maps(config)
