"""Locked MOTOR reanalysis. Outputs are aggregate; no result-selected analyses."""
from pathlib import Path
import itertools,json,math
import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats
from acquire_motor import acquire,ROOT

OUT=ROOT/'outputs/motor';TAB=ROOT/'tables';AUD=ROOT/'audit'
SPEC=json.loads((ROOT/'protocol/analysis_spec.json').read_text())

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x
def dump(path,x):path.write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf8')
def interval(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);k=len(a);l=len(b)
    mean=float(a.mean()-b.mean());v1=a.var(ddof=1)/k;v0=b.var(ddof=1)/l;se=float(np.sqrt(v1+v0))
    df=float((v1+v0)**2/(v1*v1/(k-1)+v0*v0/(l-1)))
    return dict(estimate=mean,se=se,welch_df=df,welch_ci=[mean-stats.t.ppf(.975,df)*se,mean+stats.t.ppf(.975,df)*se],
                common_df=k+l-2,common_t_ci=[mean-stats.t.ppf(.975,k+l-2)*se,mean+stats.t.ppf(.975,k+l-2)*se],
                interval_status='working-model small-sample t approximations; not design-exact confidence intervals')
def weighted(h,weight):
    risks=[];vs=[];ks=[]
    for arm in [1,2]:
        z=h[h.arm.eq(arm)];w=z[weight].to_numpy(float);r=z.risk.to_numpy(float);k=len(z)
        avg=float(np.dot(w,r)/w.sum());v=float(k/(k-1)*sum((w*(r-avg))**2)/w.sum()**2)
        risks.append(avg);vs.append(v);ks.append(k)
    e=risks[0]-risks[1];se=math.sqrt(sum(vs));df=sum(vs)**2/sum(v*v/(k-1) for v,k in zip(vs,ks))
    return dict(estimate=e,arm_risks=risks,se=se,welch_df=df,welch_ci=[e-stats.t.ppf(.975,df)*se,e+stats.t.ppf(.975,df)*se],weight=weight)
def randomization(h,metric):
    observed=float(h.loc[h.arm.eq(1),metric].mean()-h.loc[h.arm.eq(2),metric].mean());rows=[]
    for ids in itertools.combinations(range(len(h)),3):
        mask=np.array([i in ids for i in range(len(h))]);t=float(h.loc[mask,metric].mean()-h.loc[~mask,metric].mean())
        rows.append(dict(training_hospitals=','.join(str(int(x)) for x in h.loc[mask,'hospital']),statistic=t,
            observed_assignment=bool(np.array_equal(mask,h.arm.eq(1).to_numpy())),in_two_sided_tail=bool(abs(t)>=abs(observed)-SPEC['tie_tolerance'])))
    return {'statistic':observed,'assignments':len(rows),'tail_count':sum(x['in_two_sided_tail'] for x in rows),
            'p_value':sum(x['in_two_sided_tail'] for x in rows)/len(rows),'minimum_two_sided_p':2/len(rows)},pd.DataFrame(rows)
def summary(x):
    a=pd.to_numeric(x,errors='coerce').dropna()
    return dict(n=len(a),missing=int(x.isna().sum()),mean=a.mean(),sd=a.std(ddof=1),median=a.median(),q1=a.quantile(.25),q3=a.quantile(.75),minimum=a.min(),maximum=a.max())
def aggregate(d):
    h=d.groupby(['hospital','hospital_cat'],observed=True).agg(enrolled=('record_id','size'),observed=('known','sum'),deaths=('death','sum'),lost=('lost','sum'),
        prehospital_mean=('prehospital_interval','mean'),prehospital_n=('prehospital_interval','count'),referral_exit_mean=('ref_exit_interv','mean'),referral_exit_n=('ref_exit_interv','count')).reset_index().rename(columns={'hospital_cat':'arm'})
    h['risk']=h.deaths/h.observed;h['lost_risk']=h.lost/h.enrolled
    return h.sort_values('hospital').reset_index(drop=True)

def fill_extreme(h,total,maximize):
    # Extremize sum(x_h/N_h)/3 for fixed arm death total and bounded capacities.
    z=h.sort_values(['enrolled','hospital'],ascending=[maximize,True]);left=int(total);allocation={int(i):0 for i in h.hospital}
    for row in z.itertuples():
        v=min(left,int(row.lost));allocation[int(row.hospital)]=v;left-=v
    assert left==0
    effect=sum(allocation[int(r.hospital)]/r.enrolled for r in h.itertuples())/len(h)
    return effect,allocation

def missing_sensitivity(h):
    hi=h[h.arm.eq(1)];hc=h[h.arm.eq(2)];a=int(hi.lost.sum());b=int(hc.lost.sum())
    base=float((hi.deaths/hi.enrolled).mean()-(hc.deaths/hc.enrolled).mean())
    ci=float((hi.lost/hi.enrolled).mean());cc=float((hc.lost/hc.enrolled).mean())
    full_bounds=[base-cc,base+ci]
    participant_base=float(hi.deaths.sum()/hi.enrolled.sum()-hc.deaths.sum()/hc.enrolled.sum())
    participant_bounds=[participant_base-b/hc.enrolled.sum(),participant_base+a/hi.enrolled.sum()]
    grid=[]
    for qi in np.linspace(0,1,101):
        for qc in np.linspace(0,1,101):grid.append({'q_training':qi,'q_control':qc,'equal_hospital_difference':base+qi*ci-qc*cc})
    frontier=[];alloc_cache={}
    for arm,z,maxk in [(1,hi,a),(2,hc,b)]:
        for k in range(maxk+1):
            for mx in [False,True]:alloc_cache[(arm,k,mx)]=fill_extreme(z,k,mx)
    for ki in range(a+1):
        for kc in range(b+1):
            lo=base+alloc_cache[(1,ki,False)][0]-alloc_cache[(2,kc,True)][0]
            up=base+alloc_cache[(1,ki,True)][0]-alloc_cache[(2,kc,False)][0]
            frontier.append(dict(training_unknown_deaths=ki,control_unknown_deaths=kc,min_difference=lo,max_difference=up,sign_reversal_possible=up>=-1e-12,sign_reversal_all_distributions=lo>=-1e-12,
                participant_difference=participant_base+ki/hi.enrolled.sum()-kc/hc.enrolled.sum()))
    f=pd.DataFrame(frontier);zero=f[f.control_unknown_deaths.eq(0)]
    assert (f.min_difference <= f.max_difference+1e-12).all()
    possible=zero[zero.sign_reversal_possible];guaranteed=zero[zero.sign_reversal_all_distributions];pt=zero[zero.participant_difference>=0]
    first=int(possible.training_unknown_deaths.min()) if len(possible) else None
    tipping={'control_unobserved_deaths_assumed':0,'minimum_training_deaths_for_possible_sign_reversal':first,
        'minimum_training_deaths_for_all_distributions_sign_reversal':int(guaranteed.training_unknown_deaths.min()) if len(guaranteed) else None,
        'participant_weighted_minimum_training_deaths':int(pt.training_unknown_deaths.min()) if len(pt) else None,
        'witness_training_allocation':alloc_cache[(1,first,True)][1] if first is not None else None,
        'witness_equal_hospital_difference':float(possible.iloc[0].max_difference) if len(possible) else None}
    if first is not None and tipping['minimum_training_deaths_for_all_distributions_sign_reversal'] is not None:
        assert first <= tipping['minimum_training_deaths_for_all_distributions_sign_reversal']
    scenarios=[]
    for qi,qc in itertools.product(SPEC['risk_table_values'],repeat=2):scenarios.append(dict(q_training=qi,q_control=qc,equal_hospital_difference=base+qi*ci-qc*cc))
    structured=[]
    for gi,gc in itertools.product(SPEC['risk_multipliers'],repeat=2):
        ri=(hi.deaths+hi.lost*np.minimum(1,gi*hi.risk))/hi.enrolled
        rc=(hc.deaths+hc.lost*np.minimum(1,gc*hc.risk))/hc.enrolled
        structured.append(dict(gamma_training=gi,gamma_control=gc,equal_hospital_difference=float(ri.mean()-rc.mean())))
    pd.DataFrame(grid).to_csv(OUT/'missing_risk_surface.csv',index=False);f.to_csv(OUT/'missing_integer_frontier.csv',index=False)
    pd.DataFrame(scenarios).to_csv(TAB/'S2_missing_risk_scenarios.csv',index=False);pd.DataFrame(structured).to_csv(TAB/'S3_risk_multipliers.csv',index=False)
    return dict(full_cohort_base_all_lost_survive=base,equal_hospital_bounds=full_bounds,participant_weighted_bounds=participant_bounds,
        q_training_coefficient=ci,q_control_coefficient=cc,training_lost=a,control_lost=b,tipping=tipping,
        common_risk_no_reversal=all(base+q*(ci-cc)<0 for q in np.linspace(0,1,101)),bounds_status='identification bounds, not confidence intervals')

def main():
    OUT.mkdir(parents=True,exist_ok=True);TAB.mkdir(exist_ok=True);AUD.mkdir(exist_ok=True)
    raw=acquire();d,meta=pyreadstat.read_dta(raw/'Appendix 13 MOTOR TRIAL DATASET.dta',apply_value_formats=False)
    numeric=['hospital','hospital_cat','study_period','injury_outcome','age','sex','road_user_category','injury_mechanism','total_gcs','kts','prehospital_interval','ref_exit_interv','death_day','referral','rx_capacity','headinjury','musculoskeletalinjury']
    for c in numeric:
        source=d[c];converted=pd.to_numeric(source.replace('',np.nan),errors='coerce')
        if (source.notna() & source.ne('') & converted.isna()).any():raise ValueError('Undocumented nonnumeric code in '+c)
        d[c]=converted
    assert d.record_id.nunique()==len(d),'Duplicate participant identifiers: stop'
    assert not d[['hospital','hospital_cat','study_period','injury_outcome']].isna().any().any()
    assert set(d.injury_outcome.unique()) <= {1,2,4}
    assert d.groupby('hospital').hospital_cat.nunique().eq(1).all()
    assert d.hospital.nunique()==6 and d.groupby('hospital_cat').hospital.nunique().to_dict()=={1:3,2:3}
    d['known']=d.injury_outcome.isin([1,2]);d['death']=d.injury_outcome.eq(2);d['lost']=d.injury_outcome.eq(4)
    h=aggregate(d);assert h.observed.gt(0).all();h.to_csv(TAB/'table1_hospitals.csv',index=False)
    arms=d.groupby('hospital_cat').agg(enrolled=('record_id','size'),observed=('known','sum'),deaths=('death','sum'),lost=('lost','sum')).reset_index()
    arms.to_csv(TAB/'arm_counts.csv',index=False)
    schema=[{'name':c,'label':meta.column_names_to_labels.get(c),'null_count':int(d[c].isna().sum()),'value_labels':meta.variable_value_labels.get(c)} for c in meta.column_names]
    dump(AUD/'data_schema.json',schema)
    dict_frame=pd.read_csv(raw/'Appendix 10_MOTOR_Registry_Data_Dictionary.csv')
    fields=['injury_outcome','ref_exit_interv','rx_capacity','referral','death_day','resp_rate','systolic_bp']
    dict_frame[dict_frame['Variable / Field Name'].isin(fields)].fillna('').to_csv(AUD/'critical_dictionary_rows.csv',index=False)
    audit={'rows':len(d),'columns_original':len(meta.column_names),'unique_ids':d.record_id.nunique(),'hospital_count':len(h),
        'arm_counts':arms.to_dict('records'),'hospital_counts':h.to_dict('records'),'outcome_codes':d.injury_outcome.value_counts().to_dict(),
        'age':summary(d.age),'age_outside_2_80':int((~d.age.between(2,80)).sum()),
        'death_day_deceased':summary(d.loc[d.death,'death_day']),'survivor_last_contact_date_present':False,
        'prehospital':summary(d.prehospital_interval),'referral_exit':summary(d.ref_exit_interv),
        'referral_exit_crosstabs':{c:pd.crosstab(d[c].fillna(-999),d.ref_exit_interv.notna()).reset_index().to_dict('records') for c in ['referral','rx_capacity']},
        'head_or_msk_flag_absent':int((d.headinjury.ne(1)&d.musculoskeletalinjury.ne(1)).sum()),
        'head_msk_by_outcome':pd.crosstab([d.headinjury,d.musculoskeletalinjury],d.injury_outcome).reset_index().to_dict('records'),
        'gcs_availability_by_head_flag':pd.crosstab(d.headinjury,d.total_gcs.notna()).reset_index().to_dict('records'),
        'toms_availability_by_outcome':pd.crosstab(d.injury_outcome,d.total_toms.notna()).reset_index().to_dict('records'),
        'period_counts':d.study_period.value_counts().sort_index().to_dict()}
    dump(AUD/'data_audit.json',audit)
    primary=interval(h.loc[h.arm.eq(1),'risk'],h.loc[h.arm.eq(2),'risk']);primary['arm_mean_risks']=[h.loc[h.arm.eq(a),'risk'].mean() for a in [1,2]]
    exact,dist=randomization(h,'risk');dist.to_csv(OUT/'randomization_distribution.csv',index=False)
    weights={w:weighted(h,w) for w in ['observed','enrolled']};missing=missing_sensitivity(h)
    leave=[]
    for hosp in h.hospital:
        z=h[h.hospital.ne(hosp)];leave.append(dict(omitted_hospital=int(hosp),training_hospitals=int(z.arm.eq(1).sum()),control_hospitals=int(z.arm.eq(2).sum()),
            equal_hospital_difference=z.loc[z.arm.eq(1),'risk'].mean()-z.loc[z.arm.eq(2),'risk'].mean(),participant_difference=weighted(z,'observed')['estimate']))
    pd.DataFrame(leave).to_csv(TAB/'S1_leave_one_hospital_out.csv',index=False)
    cp=d.groupby(['hospital','hospital_cat','study_period']).agg(observed=('known','sum'),deaths=('death','sum')).reset_index();cp['risk']=cp.deaths/cp.observed
    cp.to_csv(TAB/'S4_hospital_periods.csv',index=False)
    per={'available':len(cp)==72 and cp.observed.gt(0).all(),'cells':len(cp),'zero_observed_cells':int(cp.observed.eq(0).sum())}
    if per['available']:
        z=cp.groupby(['hospital','hospital_cat']).risk.mean().reset_index();per['equal_hospital_difference']=z.loc[z.hospital_cat.eq(1),'risk'].mean()-z.loc[z.hospital_cat.eq(2),'risk'].mean()
    ah=aggregate(d[d.age.between(2,80)]);age_sensitivity={'n':int(d.age.between(2,80).sum()),'difference':ah.loc[ah.arm.eq(1),'risk'].mean()-ah.loc[ah.arm.eq(2),'risk'].mean()}
    process={'prehospital':interval(h.loc[h.arm.eq(1),'prehospital_mean'],h.loc[h.arm.eq(2),'prehospital_mean'])}
    process['prehospital']['exact'],pdst=randomization(h,'prehospital_mean');pdst.to_csv(OUT/'process_randomization_distribution.csv',index=False)
    process['participant_summaries']={str(a):{c:summary(d.loc[d.hospital_cat.eq(a),c]) for c in ['prehospital_interval','ref_exit_interv']} for a in [1,2]}
    process['referral_populated_hospital_mean_difference']=h.loc[h.arm.eq(1),'referral_exit_mean'].mean()-h.loc[h.arm.eq(2),'referral_exit_mean'].mean()
    process['referral_status']='Descriptive populated-record comparison only; audit branching before causal or randomization-based inference. No inferential referral analysis run.'
    missing_descriptors=[];case_mix=[]
    for groups,label,records in [(['hospital_cat','lost'],'by_arm_loss',missing_descriptors),(['hospital','hospital_cat'],'by_hospital',case_mix)]:
        for key,z in d.groupby(groups):
            row=dict(zip(groups,key));row['n']=len(z)
            for c in SPEC['baseline_descriptors']+SPEC['secondary_process_fields']:
                if c in ['sex','road_user_category','injury_mechanism']:row[c]={'levels':z[c].value_counts(dropna=False).to_dict(),'missing':int(z[c].isna().sum())}
                else:row[c]=summary(z[c])
            records.append(row)
    dump(OUT/'missingness_descriptors.json',missing_descriptors);dump(OUT/'hospital_case_mix.json',case_mix)
    d.groupby(['hospital','hospital_cat','study_period']).agg(n=('record_id','size'),lost=('lost','sum')).to_csv(TAB/'S5_missing_by_period.csv')
    # Reconstruction only, explicitly not principal inference.
    tab=pd.crosstab(d.loc[d.known,'hospital_cat'],d.loc[d.known,'death'])
    reconstruction={'complete_case_pearson_uncorrected_p':stats.chi2_contingency(tab,correction=False).pvalue,
        'prehospital_participant_mannwhitney_p':stats.mannwhitneyu(d.loc[d.hospital_cat.eq(1),'prehospital_interval'],d.loc[d.hospital_cat.eq(2),'prehospital_interval'],alternative='two-sided').pvalue,
        'binary_recode_rule':'death=0 for code 1, death=1 for code 2, missing for code 4; public Stata code agrees',
        'note':'These participant-level comparisons trace parts of the published approach and are not primary evidence in this reanalysis.'}
    dump(OUT/'original_reconstruction_numeric.json',reconstruction)
    results={'protocol_version':'1.0','primary':primary,'exact':exact,'weighted_observed':weights,'missing':missing,'leave_one_out':leave,
        'equal_period':per,'age_sensitivity':age_sensitivity,'process':process,'hospital_size':{'enrolled_cv':h.enrolled.std(ddof=1)/h.enrolled.mean(),'observed_cv':h.observed.std(ddof=1)/h.observed.mean(),'enrolled_range':[h.enrolled.min(),h.enrolled.max()],'observed_range':[h.observed.min(),h.observed.max()]}}
    dump(OUT/'results.json',results)
    print(json.dumps(clean({'audit':{k:audit[k] for k in ['rows','hospital_count','arm_counts','age_outside_2_80','head_or_msk_flag_absent']},'primary':primary,'exact':exact,'missing':missing,'process':process['prehospital']}),indent=2))
if __name__=='__main__':main()
