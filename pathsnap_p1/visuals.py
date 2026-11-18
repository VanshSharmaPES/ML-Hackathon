"""Actual-data plots and optional Folium maps (never fabricated outputs)."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})


def save_trajectory_plot(trip, destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    xy=trip.nodes[['x_m','y_m']].to_numpy()
    arcs=trip.arcs.loc[trip.arcs.valid]
    segments=xy[arcs[['from_node','to_node']].to_numpy()]
    obs=trip.observations
    lo=obs[['x_m','y_m']].min().to_numpy()-300
    hi=obs[['x_m','y_m']].max().to_numpy()+300
    visible=(segments.max(axis=1)>=lo).all(axis=1)&(segments.min(axis=1)<=hi).all(axis=1)
    fig,ax=plt.subplots(figsize=(10,7))
    ax.add_collection(LineCollection(segments[visible],colors='#bcc6cf',linewidths=.5,label='Road network'))
    if trip.route is not None:
        reference=xy[trip.arcs.iloc[trip.route][['from_node','to_node']].to_numpy()]
        ax.add_collection(LineCollection(reference,colors='#14835c',linewidths=2,label='Reference route'))
    ax.scatter(obs.x_m,obs.y_m,s=3,c='#1765be',label='GPS observations',zorder=3)
    gaps=obs.loc[obs.large_gap]
    if len(gaps):
        ax.scatter(gaps.x_m,gaps.y_m,s=45,facecolors='none',edgecolors='#df492b',label='Point after large gap',zorder=4)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='Easting (m)',ylabel='Northing (m)',
           title=f'{trip.split} / {trip.trip_id} | EPSG:{trip.audit["epsg"]}')
    ax.set_aspect('equal',adjustable='box');ax.legend(loc='best');fig.tight_layout()
    fig.savefig(destination);plt.close(fig)


def save_folium_map(trip, destination):
    import folium
    from branca.element import Element
    obs=trip.observations
    m=folium.Map(location=[float(obs.latitude.mean()),float(obs.longitude.mean())],tiles='OpenStreetMap',prefer_canvas=True)
    nodes=trip.nodes[['latitude','longitude']].to_numpy()
    road_layer=folium.FeatureGroup(name='Road network',show=True)
    # Display relevant bounding-box roads. Processing always uses the full network.
    lo=obs[['latitude','longitude']].min().to_numpy()-.01
    hi=obs[['latitude','longitude']].max().to_numpy()+.01
    arcs=trip.arcs.loc[trip.arcs.valid]
    segments=nodes[arcs[['from_node','to_node']].to_numpy()]
    visible=(segments.max(axis=1)>=lo).all(axis=1)&(segments.min(axis=1)<=hi).all(axis=1)
    # One GeoJSON layer avoids one enormous rendered template per road segment.
    network_geo={'type':'Feature','properties':{},'geometry':{'type':'MultiLineString',
        'coordinates':segments[visible][:,:,::-1].tolist()}}
    folium.GeoJson(network_geo,style_function=lambda _: {'color':'#929fa8','weight':1,'opacity':.5}).add_to(road_layer)
    road_layer.add_to(m)
    gps_layer=folium.FeatureGroup(name='Raw GPS observations')
    gps_geo={'type':'FeatureCollection','features':[
        {'type':'Feature','properties':{'observation_id':int(row.observation_id),'timestamp':float(row.timestamp)},
         'geometry':{'type':'Point','coordinates':[row.longitude,row.latitude]}}
        for row in obs.itertuples()]}
    folium.GeoJson(gps_geo,marker=folium.CircleMarker(radius=2,color='#1765be',fill=True),
        tooltip=folium.GeoJsonTooltip(fields=['observation_id','timestamp'])).add_to(gps_layer)
    gps_layer.add_to(m)
    if trip.route is not None:
        reference=folium.FeatureGroup(name='Reference route (development only)')
        route_features=[]
        for position,arc_id in enumerate(trip.route):
            arc=trip.arcs.iloc[arc_id]
            route_features.append({'type':'Feature','properties':{'route_position':position,'arc_id':int(arc_id)},
                'geometry':{'type':'LineString','coordinates':nodes[[int(arc.from_node),int(arc.to_node)]][:,::-1].tolist()}})
        folium.GeoJson({'type':'FeatureCollection','features':route_features},
            style_function=lambda _: {'color':'#14835c','weight':4},
            tooltip=folium.GeoJsonTooltip(fields=['route_position','arc_id'])).add_to(reference)
        reference.add_to(m)
    gaps=folium.FeatureGroup(name='Large time gaps')
    for row in obs.loc[obs.large_gap].itertuples():
        folium.CircleMarker([row.latitude,row.longitude],radius=7,color='#df492b',fill=False,
                            tooltip=f'Gap: {row.time_gap_s:g} s').add_to(gaps)
    gaps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(obs.latitude.min()),float(obs.longitude.min())],
                  [float(obs.latitude.max()),float(obs.longitude.max())]])
    m.get_root().html.add_child(Element('''<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:14px;border:1px solid #aaa;border-radius:6px;font:14px Arial">
    <b>PathSnap development map</b><br><span style="color:#1765be">&#9679; GPS</span><br>
    <span style="color:#929fa8">&#9473; Roads</span><br><span style="color:#14835c">&#9473; Reference route</span><br>
    <span style="color:#df492b">&#9675; Large gap</span></div>'''))
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    m.save(str(destination))


def eda_plots(inventory, sweep, observations, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    settings=[('clean_observations','GPS observations per trip'),('observed_length_m','Observed trajectory length (m)'),
              ('median_gap_s','Median sampling interval (s)'),('arcs','Road segments per trip'),
              ('large_gaps','Large gaps per trip'),('weak_components','Weak components per trip')]
    for ax,(column,title) in zip(axes.ravel(),settings):
        for split,color in [('train','#1765be'),('student_test','#de8d25')]:
            ax.hist(inventory.loc[inventory.split==split,column].dropna(),bins=15,alpha=.6,label=split,color=color)
        ax.set(title=title,ylabel='Trips');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'dataset_summary.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for split,group in observations.groupby('split'):
        axes[0].scatter(group.longitude,group.latitude,s=1,alpha=.15,label=split)
        dt=group.time_gap_s.dropna()
        axes[1].hist(np.log10(dt[dt>0]),bins=40,alpha=.5,label=split)
    axes[0].set(xlabel='Longitude',ylabel='Latitude',title='Coordinate distribution (all observations)')
    axes[1].set(xlabel='log10 time gap (seconds)',ylabel='Count',title='Sampling intervals')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(output/'coordinates_and_intervals.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,method in zip(axes,['radius','top_k']):
        for split,group in sweep.loc[sweep.method==method].groupby('split'):
            totals=group.groupby('value')[['missed','labelled_observations']].sum()
            ax.plot(totals.index,100*totals.missed/totals.labelled_observations,marker='o',label=split)
        ax.set(xlabel='Radius (m)' if method=='radius' else 'Number of candidates',ylabel='Miss rate (%)',title=method)
        ax.legend()
    fig.tight_layout();fig.savefig(output/'candidate_miss_rates.png');plt.close(fig)


def model_plots(summary, confusions, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,column,title in zip(axes,['cv_macro_f1','student_macro_f1','student_top1_accuracy'],
                              ['Group CV macro F1','Student-test candidate macro F1','Student-test observation Top-1']):
        ax.bar(summary.model,summary[column],color=['#1765be','#14835c'])
        ax.set(title=title,ylim=(0,1));ax.tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(output/'model_comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(summary.model,summary.student_log_loss,color=['#1765be','#14835c'])
    ax.set(title='Student-test candidate log loss',ylabel='Log loss (lower is better)')
    fig.tight_layout();fig.savefig(output/'model_log_loss.png');plt.close(fig)
    for model,cm in confusions.items():
        fig,ax=plt.subplots(figsize=(4,4));ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{cm[i][j]:,}',ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted label',ylabel='Reference-derived label',title=model)
        fig.tight_layout();fig.savefig(output/f'{model}_confusion.png');plt.close(fig)
