"""Independent raw-file audit: native pandas Stata parser, row loops, rational arithmetic.

Does not import src/motor_analysis.py. t critical values use numerical integration
of the density and root finding, not scipy.stats.t.ppf. Missing allocations are
exhaustively enumerated within each arm, not greedily allocated.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict,Counter
import itertools,json,math,hashlib,unittest
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
R=json.loads((ROOT/'outputs/motor/results.json').read_text())
D=pd.read_stata(ROOT/'data/raw/motor/Appendix 13 MOTOR TRIAL DATASET.dta',convert_categoricals=False)
ROWS=D.to_dict('records');H={}
for row in ROWS:
    k=int(row['hospital']);h=H.setdefault(k,dict(arm=int(row['hospital_cat']),N=0,O=0,D=0,L=0,times=[]))
    assert h['arm']==row['hospital_cat'];h['N']+=1
    h['O']+=row['injury_outcome'] in [1,2];h['D']+=row['injury_outcome']==2;h['L']+=row['injury_outcome']==4
    h['times'].append(float(row['prehospital_interval']))
for h in H.values():h['risk']=F(h['D'],h['O']);h['time']=math.fsum(h['times'])/h['N']
H=dict(sorted(H.items()));I=[k for k,v in H.items() if v['arm']==1];C=[k for k,v in H.items() if v['arm']==2]
def mean(x):return sum(x)/len(x)
def contrast(h=H,field='risk'):
    return mean([v[field] for v in h.values() if v['arm']==1])-mean([v[field] for v in h.values() if v['arm']==2])
def variance(x):
    m=mean(x);return math.fsum((float(v)-m)**2 for v in x)/(len(x)-1)
def tcrit(df):
    norm=math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    def cdf(x):return .5+quad(lambda z:norm*(1+z*z/df)**(-(df+1)/2),0,x,epsabs=1e-12)[0]
    return brentq(lambda x:cdf(x)-.975,0,50,xtol=1e-12)
def ci(a,b,weights=None):
    if weights is None:
        e=mean(a)-mean(b);v=[variance(a)/len(a),variance(b)/len(b)]
    else:
        av=[];v=[]
        for x,w in zip([a,b],weights):
            mu=sum(xx*ww for xx,ww in zip(x,w))/sum(w);av.append(mu)
            v.append(len(x)/(len(x)-1)*sum(ww**2*(xx-mu)**2 for xx,ww in zip(x,w))/sum(w)**2)
        e=av[0]-av[1]
    se=math.sqrt(sum(v));df=sum(v)**2/(v[0]**2/(len(a)-1)+v[1]**2/(len(b)-1))
    return float(e),se,df,[float(e)-tcrit(df)*se,float(e)+tcrit(df)*se]
def arm_frontier(ids):
    z=defaultdict(list)
    for allocation in itertools.product(*[range(H[k]['L']+1) for k in ids]):
        z[sum(allocation)].append(sum(F(x,H[k]['N']*3) for x,k in zip(allocation,ids)))
    return {n:(min(vals),max(vals)) for n,vals in z.items()},sum(len(x) for x in z.values())
AFI,NI=arm_frontier(I);AFC,NC=arm_frontier(C)
BASE=mean([F(H[k]['D'],H[k]['N']) for k in I])-mean([F(H[k]['D'],H[k]['N']) for k in C])

class Verify(unittest.TestCase):
    def close(self,a,b):self.assertAlmostEqual(float(a),float(b),places=10)
    def test_01_source_hash(self):
        m=json.loads((ROOT/'data/manifest.json').read_text())
        for x in m['files']:self.assertEqual(hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest(),x['sha256'])
    def test_02_unique_cohort(self):
        self.assertEqual(len(ROWS),1003);self.assertEqual(len({x['record_id'] for x in ROWS}),1003)
        self.assertEqual(len(H),6);self.assertEqual((len(I),len(C)),(3,3))
    def test_03_all_hospital_counts(self):
        t=pd.read_csv(ROOT/'tables/table1_hospitals.csv')
        for r in t.to_dict('records'):
            h=H[int(r['hospital'])]
            for column,key in [('arm','arm'),('enrolled','N'),('observed','O'),('deaths','D'),('lost','L')]:self.assertEqual(r[column],h[key])
            self.close(r['risk'],h['risk']);self.close(r['prehospital_mean'],h['time'])
    def test_04_outcomes_and_arm_counts(self):
        self.assertEqual(Counter(x['injury_outcome'] for x in ROWS),{1:805,2:82,4:116})
        for ids,want in [(I,(501,457,24,44)),(C,(502,430,58,72))]:self.assertEqual(tuple(sum(H[k][c] for k in ids) for c in ['N','O','D','L']),want)
    def test_05_primary_rational(self):self.close(contrast(),R['primary']['estimate'])
    def test_06_independent_t_intervals(self):
        for field,actual in [('risk',R['primary']),('time',R['process']['prehospital'])]:
            e,se,df,inter=ci([float(H[k][field]) for k in I],[float(H[k][field]) for k in C])
            for x,y in zip([e,se,df]+inter,[actual['estimate'],actual['se'],actual['welch_df']]+actual['welch_ci']):self.close(x,y)
            for x,y in zip([e-tcrit(4)*se,e+tcrit(4)*se],actual['common_t_ci']):self.close(x,y)
    def test_07_exact_bitmask_enumeration(self):
        vals={}
        for mask in range(64):
            if mask.bit_count()!=3:continue
            pos=[k for i,k in enumerate(H) if mask&(1<<i)];neg=[k for k in H if k not in pos]
            vals[','.join(map(str,pos))]=mean([H[k]['risk'] for k in pos])-mean([H[k]['risk'] for k in neg])
        self.assertEqual(len(vals),20)
        for row in pd.read_csv(ROOT/'outputs/motor/randomization_distribution.csv').to_dict('records'):self.close(vals[row['training_hospitals']],row['statistic'])
        tail=sum(abs(v)>=abs(contrast()) for v in vals.values())
        self.assertEqual(tail,4);self.close(F(tail,20),R['exact']['p_value'])
        for v in vals.values():self.assertIn(-v,vals.values())
    def test_08_process_enumeration(self):
        observed=contrast(field='time');values=[]
        for pos in itertools.combinations(H,3):
            neg=[k for k in H if k not in pos];values.append(mean([H[k]['time'] for k in pos])-mean([H[k]['time'] for k in neg]))
        self.assertEqual(sum(abs(v)>=abs(observed)-1e-12 for v in values),2)
        self.close(R['process']['prehospital']['exact']['p_value'],.1)
    def test_09_weights_and_uncertainty(self):
        for label,key in [('observed','O'),('enrolled','N')]:
            a=[float(H[k]['risk']) for k in I];b=[float(H[k]['risk']) for k in C]
            e,se,df,inter=ci(a,b,[[H[k][key] for k in I],[H[k][key] for k in C]])
            x=R['weighted_observed'][label]
            for p,q in zip([e,se,df]+inter,[x['estimate'],x['se'],x['welch_df']]+x['welch_ci']):self.close(p,q)
    def test_10_integer_frontier_exhaustive(self):
        self.assertEqual((NI,NC),(3105,15525))
        for row in pd.read_csv(ROOT/'outputs/motor/missing_integer_frontier.csv').to_dict('records'):
            ki=row['training_unknown_deaths'];kc=row['control_unknown_deaths'];lo=BASE+AFI[ki][0]-AFC[kc][1];hi=BASE+AFI[ki][1]-AFC[kc][0]
            self.close(lo,row['min_difference']);self.close(hi,row['max_difference'])
            self.assertEqual(lo>=0,row['sign_reversal_all_distributions']);self.assertEqual(hi>=0,row['sign_reversal_possible'])
    def test_11_tipping_thresholds_and_witness(self):
        t=R['missing']['tipping'];a=min(k for k,v in AFI.items() if BASE+v[1]>=0);b=min(k for k,v in AFI.items() if BASE+v[0]>=0)
        self.assertEqual(a,t['minimum_training_deaths_for_possible_sign_reversal']);self.assertEqual(b,t['minimum_training_deaths_for_all_distributions_sign_reversal'])
        w={int(k):v for k,v in t['witness_training_allocation'].items()};self.assertEqual(sum(w.values()),a)
        self.close(BASE+sum(F(v,3*H[k]['N']) for k,v in w.items()),t['witness_equal_hospital_difference'])
        threshold=min(k for k in range(45) if F(24+k,501)-F(58,502)>=0);self.assertEqual(threshold,34)
    def test_12_bounds(self):
        lo=BASE-AFC[72][1];hi=BASE+AFI[44][1]
        for x,y in zip([lo,hi],R['missing']['equal_hospital_bounds']):self.close(x,y)
        for x,y in zip([F(24,501)-F(130,502),F(68,501)-F(58,502)],R['missing']['participant_weighted_bounds']):self.close(x,y)
    def test_13_risk_surface_and_structured(self):
        for row in pd.read_csv(ROOT/'outputs/motor/missing_risk_surface.csv').to_dict('records'):
            q={1:row['q_training'],2:row['q_control']};x={k:{'arm':h['arm'],'r':(h['D']+q[h['arm']]*h['L'])/h['N']} for k,h in H.items()}
            self.close(contrast(x,'r'),row['equal_hospital_difference'])
        for row in pd.read_csv(ROOT/'tables/S3_risk_multipliers.csv').to_dict('records'):
            q={1:row['gamma_training'],2:row['gamma_control']};x={k:{'arm':h['arm'],'r':(h['D']+min(1,q[h['arm']]*h['risk'])*h['L'])/h['N']} for k,h in H.items()}
            self.close(contrast(x,'r'),row['equal_hospital_difference'])
    def test_14_leave_one_out(self):
        for x in R['leave_one_out']:
            h={k:v for k,v in H.items() if k!=x['omitted_hospital']};self.close(contrast(h),x['equal_hospital_difference'])
            arms=[[v for v in h.values() if v['arm']==a] for a in [1,2]]
            self.close(F(sum(v['D'] for v in arms[0]),sum(v['O'] for v in arms[0]))-F(sum(v['D'] for v in arms[1]),sum(v['O'] for v in arms[1])),x['participant_difference'])
    def test_15_period_sensitivity(self):
        cells=defaultdict(lambda:[0,0])
        for r in ROWS:
            v=cells[(int(r['hospital']),int(r['study_period']))];v[0]+=r['injury_outcome'] in [1,2];v[1]+=r['injury_outcome']==2
        self.assertEqual(len(cells),72);self.assertTrue(all(v[0]>0 for v in cells.values()))
        hh={k:{'arm':H[k]['arm'],'r':mean([F(cells[(k,p)][1],cells[(k,p)][0]) for p in range(1,13)])} for k in H}
        self.close(contrast(hh,'r'),R['equal_period']['equal_hospital_difference'])
    def test_16_age(self):
        selected=[r for r in ROWS if 2<=r['age']<=80];self.assertEqual(len(selected),1002)
        hh={}
        for k in H:
            z=[r for r in selected if r['hospital']==k];hh[k]={'arm':H[k]['arm'],'r':F(sum(r['injury_outcome']==2 for r in z),sum(r['injury_outcome'] in [1,2] for r in z))}
        self.close(contrast(hh,'r'),R['age_sensitivity']['difference'])
    def test_17_temporal_endpoint(self):
        present=lambda v:pd.notna(v) and v!=''
        deathdays=[float(r['death_day']) for r in ROWS if present(r['death_day'])]
        self.assertEqual(len(deathdays),82);self.assertEqual((min(deathdays),max(deathdays)),(0,15))
        self.assertTrue(all(present(r['death_day'])==(r['injury_outcome']==2) for r in ROWS))
    def test_18_critical_missingness(self):
        self.assertEqual(sum(pd.isna(r['ref_exit_interv']) for r in ROWS),312)
        self.assertEqual(sum(pd.isna(r['prehospital_interval']) for r in ROWS),0)
        self.assertEqual(sum(pd.isna(r['total_gcs']) for r in ROWS),54)
        self.assertEqual(sum(r['headinjury']==0 and r['musculoskeletalinjury']==0 for r in ROWS),6)
    def test_19_descriptive_supplements_and_process_counts(self):
        # Hand interpolation/row loops instead of primary pandas quantile/groupby.
        def quantile(vals,q):
            x=sorted(vals);j=(len(x)-1)*q;lo=int(math.floor(j));hi=int(math.ceil(j))
            return x[lo]+(j-lo)*(x[hi]-x[lo])
        for filename in ['missingness_descriptors','hospital_case_mix']:
            for item in json.loads((ROOT/'outputs/motor'/(filename+'.json')).read_text()):
                rr=[r for r in ROWS if r['hospital_cat']==item['hospital_cat']]
                if 'hospital' in item:rr=[r for r in rr if r['hospital']==item['hospital']]
                if 'lost' in item:rr=[r for r in rr if (r['injury_outcome']==4)==item['lost']]
                self.assertEqual(len(rr),item['n'])
                for field,summary in item.items():
                    if not isinstance(summary,dict):continue
                    vals=[float(r[field]) for r in rr if pd.notna(r[field])]
                    self.assertEqual(summary['missing'],len(rr)-len(vals))
                    if 'levels' in summary:
                        self.assertEqual({str(int(k)):n for k,n in Counter(vals).items()},summary['levels'])
                    elif vals:
                        checks={'n':len(vals),'mean':math.fsum(vals)/len(vals),'sd':math.sqrt(variance(vals)),'median':quantile(vals,.5),'q1':quantile(vals,.25),'q3':quantile(vals,.75),'minimum':min(vals),'maximum':max(vals)}
                        for key,value in checks.items():self.close(value,summary[key])
        for row in pd.read_csv(ROOT/'tables/S5_missing_by_period.csv').to_dict('records'):
            rr=[r for r in ROWS if r['hospital']==row['hospital'] and r['study_period']==row['study_period']]
            self.assertEqual(len(rr),row['n']);self.assertEqual(sum(r['injury_outcome']==4 for r in rr),row['lost'])
        for row in pd.read_csv(ROOT/'tables/S4_hospital_periods.csv').to_dict('records'):
            rr=[r for r in ROWS if r['hospital']==row['hospital'] and r['study_period']==row['study_period']]
            known=sum(r['injury_outcome'] in [1,2] for r in rr);deaths=sum(r['injury_outcome']==2 for r in rr)
            self.assertEqual(known,row['observed']);self.assertEqual(deaths,row['deaths']);self.close(F(deaths,known),row['risk'])
        for row in pd.read_csv(ROOT/'tables/S2_missing_risk_scenarios.csv').to_dict('records'):
            want=float(BASE)+row['q_training']*float(AFI[44][1])-row['q_control']*float(AFC[72][1])
            self.close(want,row['equal_hospital_difference'])
        self.assertLess(max(H[k]['time'] for k in I),min(H[k]['time'] for k in C))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Verify))
    report={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
            'independent_parser':'pandas.read_stata; main uses pyreadstat','primary_exact_fraction':str(contrast()),'primary_difference':float(contrast()),
            'exhaustive_arm_allocations':[NI,NC],'t_quantiles':'integrated Student density + Brent root, independent of scipy.stats.t.ppf'}
    (ROOT/'audit/independent_verification.json').write_text(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
