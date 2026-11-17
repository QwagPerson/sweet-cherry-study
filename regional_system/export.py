"""Validate and publish a local inference bundle, with a comparative activation gate."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
from contextlib import contextmanager
from train import ROOT,OUT,SPECS,Study,ident,metrics
from latency_calculator_app.core import regional_model as r
from latency_calculator_app.core.data import load_zone_temperature_series
from latency_calculator_app.core.normalization import normalize_zone_name

@contextmanager
def candidate_bundle(bundle):
    previous=r.load_bundle
    r.load_bundle=lambda:bundle
    try:yield
    finally:r.load_bundle=previous

def main():
    bundle=json.loads((OUT/'bundle.json').read_text(encoding='utf-8'));st=Study();x=st.x
    # Verify original feature reconstruction independently for every existing record.
    raw={};rebuilt=[]
    for row in x.itertuples():
        if row.lugar not in raw:raw[row.lugar]=load_zone_temperature_series(normalize_zone_name(row.lugar))
        rebuilt.append(r.original_feature_row(raw[row.lugar],bundle['metadata'][row.codigo_lugar],row.fecha_caida_hoja_50,row.codigo_variedad,row.codigo_tratamiento_entrada,row.codigo_tratamiento_salida))
    rebuilt=pd.DataFrame(rebuilt)
    np.testing.assert_allclose(rebuilt[r.ORIGINAL_COLD_NUMERIC],x[r.ORIGINAL_COLD_NUMERIC],atol=1e-6,rtol=1e-9,equal_nan=True)
    ref='chill_10__gdh_4_5__ridge_10';j=[ident(s) for s in SPECS].index(ref)
    oof=np.load(OUT/'validation_arrays.npz')['oof'];ids=np.arange(len(x));ctrl=x.control.to_numpy()
    reference=metrics(st,oof[ctrl,j,1],ids[ctrl])
    nested=pd.read_csv(OUT/'nested_metrics.csv');reg=nested[nested.policy.eq('regional')&nested.zone.eq('ALL')].iloc[0].to_dict()
    bundle['reference_rule']=dict(model_id=ref,selection_level='reference',validation=reference)
    bundle['reference_zones']={}
    for code in x.codigo_lugar.unique():
        ix=ids[ctrl&x.codigo_lugar.eq(code).to_numpy()]
        bundle['reference_zones'][code]=dict(model_id=ref,selection_level='reference',validation=metrics(st,oof[ix,j,1],ix))
    # No automatic promotion when regional selection loses coverage or field accuracy.
    enable=reg['valid']==reg['n'] and reference['valid']==reference['n'] and reg['mae']<reference['mae']
    bundle['deployment']=dict(regional_enabled=bool(enable),criterion='full external coverage and lower field MAE than fixed reference on the same controls',regional_mae=reg['mae'],reference_mae=reference['mae'],n_controls=reference['n'],recommended_mode='regional' if enable else 'reference',experimental_regional_available=True)
    bundle['feature_reconstruction_verified_rows']=len(x)
    bundle['context_support']={str(code)+'|'+str(var):int(len(g)) for (code,var),g in x[x.control].groupby(['codigo_lugar','codigo_variedad'])}
    # Record the final runtime code without changing historical experiment manifests.
    bundle['runtime_sha256']=hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest()
    destination=r.BUNDLE_PATH;destination.parent.mkdir(exist_ok=True,parents=True)
    from latency_calculator_app.core.prediction import predict_latency
    rows=[]
    with candidate_bundle(bundle):
        for strategy in ['recommended','regional']:
            for i in ids[ctrl]:
                row=x.iloc[i];rule,model,_=r.select_rule(bundle,row.codigo_lugar,strategy=strategy)
                expected,_=st.fit_predict(ids,np.array([i]),model['spec'])
                p=predict_latency(normalize_zone_name(row.lugar),row.variedad,row.fecha_caida_hoja_50,model_strategy=strategy)
                origin=pd.Timestamp(int(row.temporada),1,1)
                got=np.array([(pd.Timestamp(t)-origin)/pd.Timedelta(days=1) if t is not None else np.nan for t in [p.cold_end_date,p.budbreak_date]])
                np.testing.assert_allclose(got,expected[0],atol=1e-9,rtol=0,equal_nan=True)
                rows.append(dict(row_id=int(i),strategy=strategy,zone=row.codigo_lugar,year=int(row.temporada),model_id=rule['model_id'],cold_requirement=p.cold_requirement,heat_requirement=p.heat_requirement,field_date=str(p.budbreak_date),status='predicted' if p.budbreak_date is not None else 'no_crossing'))
    # Only publish after the serving path and feature reconstruction passed.
    tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(bundle,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(destination)
    r.clear_caches()
    pd.DataFrame(rows).to_csv(OUT/'runtime_replay.csv',index=False)
    rules=[]
    for code,rule in bundle['zones'].items():
        s=bundle['models'][rule['model_id']]['spec']
        rules.append(dict(zone=code,**s,selection_level=rule['selection_level'],control_support=rule['control_support'],seasons=rule['seasons'],external_mae=rule['validation']['mae'],coverage=rule['validation']['coverage']))
    pd.DataFrame(rules).to_csv(OUT/'regional_rules.csv',index=False)
    (OUT/'deployment.json').write_text(json.dumps(dict(bundle['deployment'],runtime_replay_queries=len(rows),feature_reconstruction_rows=len(x),artifact_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(json.dumps(bundle['deployment'],indent=2));print('Feature equality:',len(x),'Runtime replay:',len(rows))

if __name__=='__main__':main()
