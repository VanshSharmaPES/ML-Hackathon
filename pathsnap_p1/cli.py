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
    parser.add_argument('--results-dir',help='Separate destination for model outputs; prepared data still comes from config output_dir')
    parser.add_argument('--model',choices=['knn','gaussian_nb','logistic_regression','random_forest','hist_gradient_boosting'],
                        help='Train one model; by default train the KNN and Gaussian NB baselines')
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
        from .models import train,baseline_estimators
        estimators=None
        if args.model:
            if args.model in ('knn','gaussian_nb'):
                estimators={args.model:baseline_estimators(config)[args.model]}
            elif args.model=='random_forest':
                from sklearn.ensemble import RandomForestClassifier
                estimators={'random_forest':RandomForestClassifier(n_estimators=100,max_depth=20,min_samples_leaf=2,
                    max_features='sqrt',random_state=config['random_seed'],n_jobs=2)}
            elif args.model=='logistic_regression':
                from sklearn.linear_model import LogisticRegression
                estimators={'logistic_regression':LogisticRegression(C=1.0,max_iter=300,solver='lbfgs')}
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                estimators={'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=31,
                    learning_rate=0.1,l2_regularization=0.0,random_state=config['random_seed'])}
        results_dir=Path(args.results_dir).resolve() if args.results_dir else None
        train(config,estimators=estimators,resume=args.resume,results_dir=results_dir)
    if args.command=='maps':regenerate_maps(config)
