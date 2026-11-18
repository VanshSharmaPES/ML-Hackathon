"""Shared group-aware harness for P1 baselines; P2 can pass other estimators.

Candidate macro F1 is binary (correct/incorrect candidate). Observation Top-1
uses ALL labelled observations, including candidate misses, as its denominator.
"""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,log_loss,confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES,KEYS
from .pipeline import json_write,file_hash
from .visuals import model_plots


def make_pipeline(estimator):
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                     ('scaler',StandardScaler()),('classifier',estimator)])


def baseline_estimators(config):
    return {
        'knn':KNeighborsClassifier(n_neighbors=config['knn_neighbors'],weights=config['knn_weights'],
                                  algorithm='kd_tree',leaf_size=40,n_jobs=2),
        'gaussian_nb':GaussianNB(var_smoothing=config['gaussian_nb_var_smoothing'])}


def evaluate_predictions(frame,probabilities,observations):
    y=frame.label.to_numpy(int)
    probabilities=np.clip(np.asarray(probabilities),0,1)
    predicted=(probabilities>.5).astype(int)
    metrics={'candidate_accuracy':accuracy_score(y,predicted),
             'macro_precision':precision_score(y,predicted,labels=[0,1],average='macro',zero_division=0),
             'macro_recall':recall_score(y,predicted,labels=[0,1],average='macro',zero_division=0),
             'macro_f1':f1_score(y,predicted,labels=[0,1],average='macro',zero_division=0),
             'log_loss':log_loss(y,np.column_stack([1-probabilities,probabilities]),labels=[0,1])}
    scored=frame[KEYS+['candidate_arc_id','distance_rank']].copy()
    scored['score']=probabilities
    scored.sort_values(KEYS+['score','distance_rank','candidate_arc_id'],
                       ascending=[True,True,True,False,True,True],kind='stable',inplace=True)
    best=scored.drop_duplicates(KEYS).rename(columns={'candidate_arc_id':'predicted_arc_id'})
    labelled_obs=observations.loc[observations.reference_arc_id>=0].copy()
    result=labelled_obs.merge(best[KEYS+['predicted_arc_id','score']],on=KEYS,how='left',validate='one_to_one')
    result['correct']=result.predicted_arc_id.eq(result.reference_arc_id)
    metrics['top1_accuracy']=float(result.correct.mean())
    metrics['trip_macro_top1']=float(result.groupby(['split','trip_id']).correct.mean().mean())
    metrics['candidate_miss_rate']=float((~result.reference_in_candidates).mean())
    metrics['labelled_observations']=len(result)
    metrics['candidate_rows']=len(frame)
    return metrics,confusion_matrix(y,predicted,labels=[0,1]).tolist(),result


def load_prepared(config):
    out=Path(config['output_dir'])
    run=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    columns=KEYS+['candidate_arc_id','distance_rank','reference_arc_id','label']+FEATURES
    columns=list(dict.fromkeys(columns))
    pieces=[]
    for item in run['trips']:
        # Relative paths make the project portable; old absolute manifest fields are descriptive.
        path=out/'processed'/item['split']/item['trip_id']/'candidates.parquet'
        pieces.append(pd.read_parquet(path,columns=columns))
    frame=pd.concat(pieces,ignore_index=True)
    observations=pd.read_parquet(out/'processed'/'observations.parquet')
    frame=frame.loc[frame.label>=0].reset_index(drop=True)
    return frame,observations,run


def experiment_signature(config, estimators):
    """Signature excludes maps/report metadata; binds metrics to data and model settings."""
    relevant={k:v for k,v in config.items() if k not in ('data_root','output_dir','maps_per_partition')}
    return {'config':relevant,'source_manifest_sha256':file_hash(Path(config['output_dir'])/'source_manifest.csv'),
            'features':FEATURES,'parameters':{name:e.get_params() for name,e in estimators.items()}}


def train(config, estimators=None, resume=False, results_dir=None):
    source_out=Path(config['output_dir'])
    out=Path(results_dir) if results_dir is not None else source_out
    (out/'models').mkdir(parents=True,exist_ok=True)
    (out/'predictions').mkdir(parents=True,exist_ok=True)
    frame,observations,run=load_prepared(config)
    train_df=frame.loc[frame.split=='train'].reset_index(drop=True)
    test_df=frame.loc[frame.split=='student_test'].reset_index(drop=True)
    train_obs=observations.loc[observations.split=='train']
    test_obs=observations.loc[observations.split=='student_test']
    if train_df.label.nunique()!=2 or test_df.empty:
        raise ValueError('Need both training classes and a nonempty student-test partition.')
    folds=min(config['cv_folds'],train_df.trip_id.nunique())
    if folds<2:raise ValueError('At least two training trips needed for grouped CV.')
    splitter=GroupKFold(n_splits=folds)
    splits=list(splitter.split(train_df,train_df.label,groups=train_df.trip_id))
    split_record=[]
    for i,(tr,va) in enumerate(splits):
        split_record.append({'fold':i+1,'train_trips':sorted(train_df.iloc[tr].trip_id.unique().tolist()),
                             'validation_trips':sorted(train_df.iloc[va].trip_id.unique().tolist())})
    estimators=baseline_estimators(config) if estimators is None else estimators
    signature=experiment_signature(config,estimators)
    checkpoint=out/'experiment_signature.json'
    prior=pd.DataFrame()
    if resume:
        if not checkpoint.exists() or json.loads(checkpoint.read_text())!=signature:
            raise ValueError('Resume rejected: dataset/config/model signature does not match.')
        if json.loads((out/'cv_splits.json').read_text())!=split_record:
            raise ValueError('Resume rejected: group folds changed.')
        if (out/'cv_results.csv').exists():prior=pd.read_csv(out/'cv_results.csv')
    json_write(out/'cv_splits.json',split_record)
    json_write(checkpoint,signature)
    all_folds=[];summaries=[];confusions={};details={}
    X=train_df[FEATURES];y=train_df.label.to_numpy(int)
    for name,estimator in estimators.items():
        print(f'Training {name}: {len(train_df):,} rows; {folds} trip-grouped folds',flush=True)
        pipeline=make_pipeline(estimator)
        model_fold=[]
        for fold,(tr,va) in enumerate(splits,1):
            cached=prior.loc[(prior.model==name)&(prior.fold==fold)] if len(prior) else prior
            if len(cached)==1:
                metrics=cached.iloc[0].to_dict();model_fold.append(metrics);all_folds.append(metrics)
                print(f'  fold {fold}: reused verified completed metrics',flush=True)
                continue
            model=clone(pipeline)
            start=time.perf_counter();model.fit(X.iloc[tr],y[tr]);fit_s=time.perf_counter()-start
            start=time.perf_counter();proba=model.predict_proba(X.iloc[va])[:,1];predict_s=time.perf_counter()-start
            val=train_df.iloc[va]
            val_obs=train_obs.loc[train_obs.trip_id.isin(val.trip_id.unique())]
            metrics,_,_=evaluate_predictions(val,proba,val_obs)
            metrics.update(model=name,fold=fold,training_seconds=fit_s,inference_seconds=predict_s)
            model_fold.append(metrics);all_folds.append(metrics)
            print(f'  fold {fold}: macro F1={metrics["macro_f1"]:.4f}, Top-1={metrics["top1_accuracy"]:.4f}; fit {fit_s:.1f}s, infer {predict_s:.1f}s',flush=True)
            pd.DataFrame(all_folds).to_csv(out/'cv_results.csv',index=False)
        model=clone(pipeline)
        start=time.perf_counter();model.fit(X,y);fit_s=time.perf_counter()-start
        start=time.perf_counter();test_proba=model.predict_proba(test_df[FEATURES])[:,1];test_s=time.perf_counter()-start
        metrics,cm,point_predictions=evaluate_predictions(test_df,test_proba,test_obs)
        # Resubstitution is diagnostic only, never the selection metric.
        # A fixed observation sample keeps resubstitution diagnostic affordable.
        # Model fitting, CV and student evaluation still use every candidate row.
        diagnostic_obs=train_obs.sample(n=min(2500,len(train_obs)),random_state=config['random_seed'])
        diagnostic=train_df.merge(diagnostic_obs[KEYS],on=KEYS,how='inner',validate='many_to_one')
        start=time.perf_counter();train_proba=model.predict_proba(diagnostic[FEATURES])[:,1]
        train_metrics,_,_=evaluate_predictions(diagnostic,train_proba,diagnostic_obs)
        metrics['resubstitution_observations']=len(diagnostic_obs)
        metrics['resubstitution_prediction_seconds']=time.perf_counter()-start
        fold_frame=pd.DataFrame(model_fold)
        row=dict(model=name,cv_macro_f1=fold_frame.macro_f1.mean(),cv_macro_f1_std=fold_frame.macro_f1.std(ddof=0),
                 cv_top1_accuracy=fold_frame.top1_accuracy.mean(),cv_log_loss=fold_frame.log_loss.mean(),
                 train_macro_f1=train_metrics['macro_f1'],student_macro_f1=metrics['macro_f1'],
                 student_precision=metrics['macro_precision'],student_recall=metrics['macro_recall'],
                 student_candidate_accuracy=metrics['candidate_accuracy'],student_top1_accuracy=metrics['top1_accuracy'],
                 student_trip_macro_top1=metrics['trip_macro_top1'],student_log_loss=metrics['log_loss'],
                 student_candidate_miss_rate=metrics['candidate_miss_rate'],training_seconds=fit_s,
                 student_inference_seconds=test_s,cv_training_seconds=fold_frame.training_seconds.sum(),
                 training_rows=len(train_df),student_rows=len(test_df),diagnostic_observations=len(diagnostic_obs))
        summaries.append(row);confusions[name]=cm
        artifact={'pipeline':model,'features':FEATURES,'configuration':config,
                  'label_definition':'Agreement with monotonic reference-route derived labels',
                  'prepare_manifest_sha256':file_hash(source_out/'run_manifest.json')}
        joblib.dump(artifact,out/'models'/f'{name}.joblib',compress=3)
        scored=test_df[KEYS+['candidate_arc_id','reference_arc_id','label']].copy()
        scored['probability_correct']=test_proba
        scored.to_parquet(out/'predictions'/f'{name}_student_candidates.parquet',index=False)
        point_predictions.to_parquet(out/'predictions'/f'{name}_student_observations.parquet',index=False)
        point_predictions.groupby(['split','trip_id']).agg(observations=('correct','size'),top1_accuracy=('correct','mean')).to_csv(out/f'{name}_per_trip.csv')
        details[name]={'parameters':estimator.get_params(),'student_metrics':metrics,'train_metrics':train_metrics,'confusion_matrix':cm}
        json_write(out/'model_details.json',details)
        pd.DataFrame(summaries).to_csv(out/'model_comparison.csv',index=False)
        print(f'  Student test: Top-1={metrics["top1_accuracy"]:.4f}, macro F1={metrics["macro_f1"]:.4f}',flush=True)
    summary=pd.DataFrame(summaries)
    model_plots(summary,confusions,out/'figures')
    write_model_report(summary,details,run,out)
    return summary


def write_model_report(summary,details,run,out):
    model_names=[str(name) for name in summary.model.tolist()]
    is_p1_baselines=set(model_names)=={'knn','gaussian_nb'}
    title='# P1 model report — KNN and Gaussian Naive Bayes' if is_p1_baselines else '# Model report — '+', '.join(model_names)
    intro=('Actual experiments, not placeholder results. These are two of the team\'s minimum five models. P2 must add the remaining models and Bayesian optimisation; P3 must evaluate HMM/Viterbi.'
           if is_p1_baselines else 'Actual experiments from the shared training and evaluation harness. Scores use labels derived by monotonic alignment to the supplied reference routes; they are not independently verified GPS ground truth.')
    lines=[title,'',intro,'',
           '## Shared experimental design','',
           f'- Training trips: {sum(t["split"]=="train" for t in run["trips"])}; student-test trips: {sum(t["split"]=="student_test" for t in run["trips"])}.',
           '- GroupKFold keeps all points/candidates from each trip together. Median imputation and StandardScaler are fitted within each fold, then refitted on training only for final models.',
           '- All generated labelled candidate rows are used; no negative downsampling or hidden reference-candidate injection.',
           '- Candidate macro F1 averages the correct/incorrect classes. Observation Top-1 selects one road by score; ties use distance rank then arc ID. Candidate misses remain in the observation denominator.',
           '- Scores measure agreement with geometry-derived reference alignment. This creates label bias, especially for distance-based methods; it is not independently verified ground truth.',
           '- Model inference timings exclude upstream candidate generation and route decoding. Training and prediction code are shared in pathsnap_p1/models.py.',
           '- Resubstitution diagnostics use a fixed random sample of up to 2,500 training observations (all candidates for each); fitting, cross-validation and student testing use all rows.',
           '', '| Model | CV macro F1 | CV SD | Student macro F1 | Student Top-1 | Log loss | Inference seconds |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.model} | {r.cv_macro_f1:.4f} | {r.cv_macro_f1_std:.4f} | {r.student_macro_f1:.4f} | {r.student_top1_accuracy:.4f} | {r.student_log_loss:.4f} | {r.student_inference_seconds:.2f} |')
    explanations={
        'knn':('Instance-based','Similar point/candidate feature patterns may have similar labels; distance and heading interactions need not be linear.',
               'Flexible local decisions, but sensitive to scaling and noisy labels. Storage and inference cost increase with training size. Distance-weighted voting can produce overconfident probabilities.',
               'No iterative loss is minimised. Training stores transformed examples; prediction uses distance-weighted neighbour votes. Training-set self-neighbours make resubstitution performance optimistic.'),
        'gaussian_nb':('Probabilistic','A fast probabilistic baseline tests whether simple class-conditional feature distributions can distinguish candidate roads.',
                       'Fast and compact. Conditional independence and Gaussian assumptions are imperfect: distance, rank, heading and previous-distance features can be correlated or skewed. Probabilities may be poorly calibrated.',
                       'Estimates class priors and per-class feature means/variances; predicts by the Gaussian naive Bayes likelihood rule. Variance smoothing stabilises calculations.'),
        'random_forest':('Tree ensemble','A Random Forest can model nonlinear interactions among distance, heading, speed and road-context features.',
                         'Handles nonlinear feature effects and provides a useful ensemble baseline. A large forest can use substantial memory and prediction time; probabilities may require calibration.',
                         'Fits bootstrap decision trees and aggregates their class probabilities. The trees reduce variance compared with a single decision tree.'),
        'logistic_regression':('Linear probabilistic','Logistic Regression provides a fast, interpretable linear baseline for candidate correctness.',
                               'Efficient and produces probabilities, but its linear decision boundary may miss nonlinear interactions among road features.',
                               'Minimises regularised logistic loss; probabilities come from the fitted logistic function.'),
        'hist_gradient_boosting':('Boosting ensemble','HistGradientBoosting can capture nonlinear feature relationships while using histogram-based split search.',
                                  'Models nonlinear interactions and can be efficient on large tables. Results depend on tree depth, learning rate and iteration count; probability calibration may vary.',
                                  'Adds shallow decision trees sequentially to reduce logistic loss; histogram bins speed up split search.')}
    for r in summary.itertuples():
        family,why,limits,objective=explanations.get(r.model,('Custom','Document suitability.','Document limitations.','Document objective.'))
        lines+=['',f'## 3.2 — {r.model}','',f'**Family and suitability:** {family}. {why}',
                '',f'**Strengths and limitations:** {limits}','',
                '**Configuration:** `'+json.dumps(details[r.model]['parameters'],sort_keys=True)+'`','',
                f'**Objective / training rule:** {objective}','',
                f'**Measured results:** CV macro F1 {r.cv_macro_f1:.4f} (fold SD {r.cv_macro_f1_std:.4f}); student-test macro precision {r.student_precision:.4f}, macro recall {r.student_recall:.4f}, macro F1 {r.student_macro_f1:.4f}, Top-1 {r.student_top1_accuracy:.4f}, log loss {r.student_log_loss:.4f}.',
                '',f'**Generalisation evidence:** sampled training resubstitution macro F1 {r.train_macro_f1:.4f} ({r.diagnostic_observations:,} observations), CV {r.cv_macro_f1:.4f}, student-test {r.student_macro_f1:.4f}. Sampled training minus CV gap is {r.train_macro_f1-r.cv_macro_f1:.4f}; discuss this alongside fold variability and per-trip failures. A single gap does not prove overfitting; low scores alone do not prove underfitting.',
                '',f'**Efficiency:** final fit {r.training_seconds:.2f} seconds; student prediction {r.student_inference_seconds:.2f} seconds over {r.student_rows:,} candidate rows.',
                '',f'**Diagnostics:** see figures/{r.model}_confusion.png and {r.model}_per_trip.csv. These models do not produce epoch-wise histories, so no artificial training-loss curves are supplied.']
    comparison_text=('Use the measured CV and student-test table to compare these two baselines. Probability quality and Top-1 accuracy both matter for subsequent decoding. No team-wide final model is selected here because the other three or more model experiments are outside P1. No Bayesian optimisation or HMM results are claimed.'
                     if is_p1_baselines else 'Use the measured cross-validation and student-test metrics to compare these experiments. Model selection should consider generalisation, fold stability, probability quality, inference speed, route continuity and compatibility with the decoder. These results alone do not include Bayesian optimisation or HMM/Viterbi decoding.')
    lines+=['','## Model comparison interpretation','',comparison_text,'',
            'Inspect the per-trip CSVs and maps before writing team conclusions. All members should verify and understand this generated results draft before submission.']
    (out/'MODEL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
