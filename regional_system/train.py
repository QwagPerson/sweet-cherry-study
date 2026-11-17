from pathlib import Path
import sys,json,hashlib,itertools
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'sweet-cherry-front'),str(ROOT/'sweet-cherry-study')]
from latency_calculator_app.core import regional_model as r
from latency_calculator_app.core.data import load_zone_temperature_series
from latency_calculator_app.core.normalization import normalize_zone_name
from src.evaluation.grouped_cv import make_validation_folds

OUT=ROOT/'output/regional_requirement_system'
SPECS=[dict(cold=c,heat=h,alpha=a) for c,h,a in itertools.product(r.COLD_MODELS,r.HEAT_BASES,[1.,10.,100.])]
def ident(s):return f"{s['cold']}__{s['heat']}__ridge_{s['alpha']:g}"
STRATEGY='GroupKFold_lugar_variedad_temporada'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

class Study:
    def __init__(self):
        self.x=pd.read_csv(ROOT/'sweet-cherry-study/data/processed/modeling_dataset.csv')
        for k in ['fecha_caida_hoja_50','fecha_brotacion_camara_50','fecha_brotacion_campo_50']:self.x[k]=pd.to_datetime(self.x[k])
        self.x['row_id']=np.arange(len(self.x));self.x['control']=self.x.codigo_tratamiento_entrada.eq('ECON')&self.x.codigo_tratamiento_salida.eq('SCON')
        assert len(self.x)==721 and not self.x.duplicated(['unidad_experimental_id','lugar']).any()
        self.raw={};self.hist={};self.c={};self.h={};self.cstart={};self.features=[]
        for place,year in self.x[['lugar','temporada']].drop_duplicates().itertuples(index=False,name=None):
            raw=load_zone_temperature_series(normalize_zone_name(place))
            self.hist[place,year]=r.history_features(raw,year)
            now=raw[raw.ds.dt.year.eq(year)].reset_index(drop=True)
            idx=pd.date_range(f'{year}-01-01',f'{year}-12-31 23:00',freq='h')
            assert np.array_equal(now.ds.to_numpy().astype('datetime64[ns]'),idx.to_numpy()) and not now.source.eq('forecast').any()
            self.raw[place,year]=now
            for cf in r.COLD_MODELS:self.c[place,year,cf]=np.r_[0,np.cumsum(r.cold_rates(now.temp_c,cf))]
            for hf in r.HEAT_BASES:self.h[place,year,hf]=np.r_[0,np.cumsum(r.heat_rates(now.temp_c,hf))]
        self.zones={code:g.iloc[0][['latitud','longitud','condicion']].to_dict() for code,g in self.x.groupby('codigo_lugar')}
        for row in self.x.itertuples():
            self.features.append(r.feature_row(self.zones[row.codigo_lugar],self.hist[row.lugar,row.temporada],row.fecha_caida_hoja_50,row.codigo_variedad,row.codigo_tratamiento_entrada,row.codigo_tratamiento_salida))
        # Preserve the current Ridge feature specification; reconstruct it at runtime.
        self.f=self.x.copy();self.n=len(self.x)
        self.y=np.array([self.hours(row.fecha_brotacion_campo_50,row.temporada)/24 for row in self.x.itertuples()])
        self.camera=np.array([self.hours(row.fecha_brotacion_camara_50,row.temporada)/24 for row in self.x.itertuples()])
        self.targets={}
        for cf,hf in itertools.product(r.COLD_MODELS,r.HEAT_BASES):
            q=[]
            for row in self.x.itertuples():
                a=int(np.ceil(self.hours(row.fecha_caida_hoja_50,row.temporada)))
                b=int(np.floor(self.hours(row.fecha_brotacion_camara_50,row.temporada)))+1
                e=int(np.floor(self.hours(row.fecha_brotacion_campo_50,row.temporada)))+1
                cc=self.c[row.lugar,row.temporada,cf];hc=self.h[row.lugar,row.temporada,hf]
                q.append([cc[b]-cc[a],hc[e]-hc[b]])
            self.targets[cf,hf]=np.array(q)
    @staticmethod
    def hours(date,year):return (pd.Timestamp(date)-pd.Timestamp(year,1,1))/pd.Timedelta(hours=1)
    def cold_cross(self,ids,q,cf):
        dates=[]
        for i,v in zip(ids,q):
            row=self.x.iloc[i];s=int(np.ceil(self.hours(row.fecha_caida_hoja_50,row.temporada)))
            key=(row.lugar,row.temporada,s,cf)
            if key not in self.cstart:
                cc=self.c[row.lugar,row.temporada,cf]
                self.cstart[key]=np.maximum.accumulate(cc[s+1:]-cc[s])
            clock=self.cstart[key];j=np.searchsorted(clock,v)
            dates.append((s+j)/24 if j<len(clock) else np.nan)
        return np.array(dates)
    def heat_cross(self,ids,ca,q,hf):
        dates=[]
        for i,start,v in zip(ids,ca,q):
            if not np.isfinite(start):dates.append(np.nan);continue
            row=self.x.iloc[i];clock=self.h[row.lugar,row.temporada,hf]
            s=int(round(start*24))+1
            if s>=len(clock)-1:dates.append(np.nan);continue
            j=np.searchsorted(clock[s+1:],clock[s]+v)+s
            dates.append(j/24 if j<len(clock)-1 else np.nan)
        return np.array(dates)
    def fit_predict(self,tr,te,spec,export=False):
        cf,hf,alpha=spec['cold'],spec['heat'],spec['alpha'];q=self.targets[cf,hf]
        if not (q[tr]>0).all():return np.full((len(te),2),np.nan),None
        cold=r.fit_regressor(self.f.iloc[tr],q[tr,0],alpha)
        cq=r.infer(self.f.iloc[te],cold);ca=self.cold_cross(te,cq,cf)
        ht=self.f.iloc[tr].copy();ht['fecha_fin_frio_dia_del_ano']=self.x.iloc[tr].fecha_brotacion_camara_50.dt.dayofyear.to_numpy()
        ht['fecha_fin_frio_dias_desde_1_abril']=[(rr.fecha_brotacion_camara_50-pd.Timestamp(rr.temporada,4,1))/pd.Timedelta(days=1) for rr in self.x.iloc[tr].itertuples()]
        heat=r.fit_regressor(ht,q[tr,1],alpha,heat=True)
        he=self.f.iloc[te].copy();he['fecha_fin_frio_dia_del_ano']=np.floor(ca)+1
        he['fecha_fin_frio_dias_desde_1_abril']=[v-(pd.Timestamp(int(year),4,1)-pd.Timestamp(int(year),1,1))/pd.Timedelta(days=1) for v,year in zip(ca,self.x.iloc[te].temporada)]
        hq=r.infer(he,heat);field=self.heat_cross(te,ca,hq,hf)
        model={'spec':spec,'cold_regressor':cold,'heat_regressor':heat} if export else None
        return np.column_stack([ca,field]),model

def losses(p,y):
    v=np.abs(p-y[:,None]);return np.where(np.isfinite(v),v,np.inf)
def select(study,loss,train,targets,regional):
    x=study.x;control=x.control.to_numpy();base=train[control[train]]
    global_score=loss[base].mean(0);global_id=int(np.argmin(global_score))
    chosen=np.repeat(global_id,len(targets));levels=np.repeat('global',len(targets)).astype(object)
    for zone in x.iloc[targets].codigo_lugar.unique():
        mask=x.iloc[targets].codigo_lugar.eq(zone).to_numpy()
        local=base[x.iloc[base].codigo_lugar.eq(zone).to_numpy()]
        if regional and len(local)>=8 and x.iloc[local].temporada.nunique()>=2:
            chosen[mask]=np.argmin(loss[local].mean(0));levels[mask]='zone'
    return chosen,levels

def metrics(study,p,ids):
    e=p-study.y[ids];good=np.isfinite(e)
    return dict(n=len(ids),valid=int(good.sum()),coverage=float(good.mean()),mae=float(np.mean(np.abs(e[good]))) if good.any() else None,bias=float(np.mean(e[good])) if good.any() else None)

def run():
    OUT.mkdir(exist_ok=True);st=Study();x=st.x;n=st.n;ids=np.arange(n);ns=len(SPECS)
    folds=make_validation_folds(x,STRATEGY);oof=np.full((n,ns,2),np.nan);nested=np.full((len(folds),n,ns),np.nan)
    for j,spec in enumerate(SPECS):
        for k,(tr,te) in enumerate(folds):
            oof[te,j],_=st.fit_predict(tr,te,spec)
            for itr,ite in make_validation_folds(x.iloc[tr].reset_index(drop=True),STRATEGY):
                pred,_=st.fit_predict(tr[itr],tr[ite],spec)
                nested[k,tr[ite],j]=pred[:,1]
        if j%6==5:print('Candidates',j+1,'/',ns,flush=True)
    np.savez_compressed(OUT/'validation_arrays.npz',oof=oof,nested=nested)
    loss=losses(oof[:,:,1],st.y);external={};audit=[]
    for regional in [False,True]:
        pred=np.full((n,2),np.nan);candidate=np.empty(n,int);level=np.empty(n,object)
        for k,(tr,te) in enumerate(folds):
            ch,lv=select(st,losses(nested[k],st.y),tr,te,regional)
            candidate[te]=ch;level[te]=lv;pred[te]=oof[te,ch]
            for i in te:audit.append(dict(policy='regional' if regional else 'global',outer_fold=k,row_id=int(i),candidate=ident(SPECS[candidate[i]]),level=level[i]))
        external['regional' if regional else 'global']=pred
    pd.DataFrame(audit).to_csv(OUT/'selection_audit.csv',index=False)
    cv=[]
    for policy,p in external.items():
        for zone in ['ALL',*sorted(x.codigo_lugar.unique())]:
            ix=ids[x.control.to_numpy() & ((x.codigo_lugar.eq(zone).to_numpy()) if zone!='ALL' else True)]
            cv.append(dict(policy=policy,zone=zone,**metrics(st,p[ix,1],ix)))
    pd.DataFrame(cv).to_csv(OUT/'nested_metrics.csv',index=False)
    pd.DataFrame([dict(candidate=ident(s),**metrics(st,oof[x.control.to_numpy(),j,1],ids[x.control.to_numpy()])) for j,s in enumerate(SPECS)]).to_csv(OUT/'fixed_candidate_metrics.csv',index=False)
    # Full-season external test: internal selection has only the other winter.
    temporal=[]
    for year in [2024,2025]:
        tr=ids[x.temporada.ne(year)];te=ids[x.temporada.eq(year)];inner=np.full((n,ns),np.nan);outer=np.full((len(te),ns),np.nan)
        for j,spec in enumerate(SPECS):
            for itr,ite in make_validation_folds(x.iloc[tr].reset_index(drop=True),STRATEGY):
                p,_=st.fit_predict(tr[itr],tr[ite],spec);inner[tr[ite],j]=p[:,1]
            p,_=st.fit_predict(tr,te,spec);outer[:,j]=p[:,1]
        ch,lv=select(st,losses(inner,st.y),tr,te,True);p=outer[np.arange(len(te)),ch]
        for i,pp,j in zip(te,p,ch):temporal.append(dict(row_id=int(i),year=year,candidate=ident(SPECS[j]),prediction=pp,truth=st.y[i],control=bool(x.control.iloc[i])))
        print('Season holdout',year,'finished',flush=True)
    pd.DataFrame(temporal).to_csv(OUT/'season_holdout.csv',index=False)
    ch,levels=select(st,loss,ids,ids,True);global_ch,_=select(st,loss,ids,ids,False)
    global_id=int(global_ch[0]);models={}
    for spec in SPECS:
        _,model=st.fit_predict(ids,ids,spec,True)
        if model is not None:models[ident(spec)]=model
    zone_rules={}
    for zone in sorted(x.codigo_lugar.unique()):
        zi=ids[x.codigo_lugar.eq(zone)];i=zi[0];choice=ident(SPECS[ch[i]])
        record=next(v for v in cv if v['policy']=='regional' and v['zone']==zone)
        zone_rules[zone]=dict(model_id=choice,selection_level=str(levels[i]),validation=record,control_support=int(x.iloc[zi].control.sum()),seasons=int(x.iloc[zi][x.iloc[zi].control].temporada.nunique()))
    global_rule=dict(model_id=ident(SPECS[global_id]),selection_level='global',validation=next(v for v in cv if v['policy']=='global' and v['zone']=='ALL'))
    pair_models={}
    for cf,hf in itertools.product(r.COLD_MODELS,r.HEAT_BASES):
        jlist=[j for j,s in enumerate(SPECS) if s['cold']==cf and s['heat']==hf and ident(s) in models]
        if jlist:
            best=min(jlist,key=lambda j:(loss[x.control.to_numpy(),j].mean(),ident(SPECS[j])))
            pair_models[cf+'__'+hf]=ident(SPECS[best])
    files=[Path(__file__),Path(__file__).with_name('PROTOCOL.md'),Path(r.__file__),ROOT/'sweet-cherry-study/data/processed/modeling_dataset.csv']
    files+=list((ROOT/'sweet-cherry-front/data/temperature_by_zone').rglob('temperatures.*'))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    bundle=dict(schema_version=1,version='regional-ridge-20261109',created_utc=datetime.now(timezone.utc).isoformat(),training_years=[2024,2025],training_rows=n,
        metric_scope='Nested grouped validation of selection procedure; control only; retrospective weather; not independent new-season validation.',
        zones=zone_rules,global_rule=global_rule,models=models,pair_models=pair_models,metadata=st.zones,
        varieties=sorted(x.codigo_variedad.unique()),sources=manifest)
    save('bundle.json',bundle)
    rows=[]
    for policy,p in external.items():
        dd=x[['row_id','temporada','codigo_lugar','lugar','codigo_variedad','control','fecha_caida_hoja_50','fecha_brotacion_campo_50']].copy()
        dd['policy']=policy;dd['pred_camera_day']=p[:,0];dd['pred_field_day']=p[:,1];dd['error_days']=p[:,1]-st.y
        rows.append(dd)
    pd.concat(rows).to_csv(OUT/'external_predictions.csv',index=False)
    st.f.assign(row_id=ids).to_csv(OUT/'training_features.csv',index=False)
    save('source_manifest.json',manifest)
    print(pd.DataFrame(cv).to_string(index=False),flush=True)
    print('Final rules',json.dumps({z:v['model_id'] for z,v in zone_rules.items()}),flush=True)

if __name__=='__main__':run()
