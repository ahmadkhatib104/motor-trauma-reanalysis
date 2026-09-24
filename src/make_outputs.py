"""Publication plots and aggregate tables from locked analysis outputs only."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'figures';TAB=ROOT/'tables';OUT=ROOT/'outputs/motor'
R=json.loads((OUT/'results.json').read_text());A=json.loads((ROOT/'audit/data_audit.json').read_text());H=pd.read_csv(TAB/'table1_hospitals.csv')
INK='#202020';MUTED='#666666';RULE='#B5B5B5';LIGHT='#E6E6E6';SHADE='#F2F2F2'
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':INK,'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'MOTOR-locked-v1','savefig.facecolor':'white'})
MANIFEST=[]
def save(fig,name,inputs):
    FIG.mkdir(exist_ok=True)
    for ext in ['png','svg','tiff']:
        p=FIG/(name+'.'+ext)
        options={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else ({'metadata':{'Date':None}} if ext=='svg' else {})
        fig.savefig(p,dpi=600 if ext=='tiff' else 300,bbox_inches='tight',**options)
        if ext=='svg':p.write_bytes(('\n'.join(line.rstrip() for line in p.read_text(encoding='utf8').splitlines())+'\n').encode('utf8'))
        MANIFEST.append({'file':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_files':inputs,'generation':'matplotlib; plotted values derived directly from source aggregates'})
    plt.close(fig)
def md_table(df,path):
    def s(v):return str(v).replace('|','/').replace('\n',' ')
    txt='| '+' | '.join(map(s,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
    txt+='\n'.join('| '+' | '.join(s(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'
    path.write_text(txt,encoding='utf8')
def tables():
    case=json.loads((OUT/'hospital_case_mix.json').read_text());lookup={int(x['hospital']):x for x in case}
    t=[]
    for r in H.itertuples():
        t.append({'Hospital':'H'+str(r.hospital),'Arm':'Training' if r.arm==1 else 'Control','Enrolled':r.enrolled,'Known':r.observed,'Deaths':int(r.deaths),'Lost, n (%)':f'{r.lost} ({100*r.lost/r.enrolled:.1f})','Mortality, %':f'{100*r.risk:.2f}','Age, mean (SD), y':f'{lookup[r.hospital]["age"]["mean"]:.1f} ({lookup[r.hospital]["age"]["sd"]:.1f})','KTS, mean (SD)':f'{lookup[r.hospital]["kts"]["mean"]:.2f} ({lookup[r.hospital]["kts"]["sd"]:.2f})'})
    pd.DataFrame(t).to_csv(TAB/'table1_display.csv',index=False);md_table(pd.DataFrame(t),TAB/'table1_display.md')
    p=R['primary'];w=R['weighted_observed'];r=[]
    for label,v,ci in [('Equal-hospital (primary)',p,p['welch_ci']),('Equal-hospital, common-variance interval',p,p['common_t_ci']),('Participant-weighted observed risk',w['observed'],w['observed']['welch_ci']),('Enrollment-weighted observed risk',w['enrolled'],w['enrolled']['welch_ci'])]:
        r.append({'Analysis':label,'Difference, pp':f'{100*v["estimate"]:.2f}','95% interval, pp':f'{100*ci[0]:.2f} to {100*ci[1]:.2f}','Interval basis':'Common-variance t, df=4' if 'common' in label else f'Welch t, df={v["welch_df"]:.2f}'})
    pd.DataFrame(r).to_csv(TAB/'table2_primary.csv',index=False);md_table(pd.DataFrame(r),TAB/'table2_primary.md')
    s=[{'Analysis':'Equal-period hospital risks','Difference / range, pp':f'{100*R["equal_period"]["equal_hospital_difference"]:.2f}','Interpretation':'Descriptive standardization; 72 cells, still six hospitals'},
       {'Analysis':'Recorded age 2-80','Difference / range, pp':f'{100*R["age_sensitivity"]["difference"]:.2f}','Interpretation':'1002 records; eligibility sensitivity'},
       {'Analysis':'Leave one hospital out','Difference / range, pp':f'{100*min(x["equal_hospital_difference"] for x in R["leave_one_out"]):.2f} to {100*max(x["equal_hospital_difference"] for x in R["leave_one_out"]):.2f}','Interpretation':'Diagnostic, not a confidence interval or six tests'},
       {'Analysis':'All lost survive, full-cohort hospitals','Difference / range, pp':f'{100*R["missing"]["full_cohort_base_all_lost_survive"]:.2f}','Interpretation':'Different denominator from primary observed-risk contrast'},
       {'Analysis':'Equal-hospital extreme bounds','Difference / range, pp':' to '.join(f'{100*x:.2f}' for x in R['missing']['equal_hospital_bounds']),'Interpretation':'Identification bounds; not confidence intervals'},
       {'Analysis':'Participant-weighted extreme bounds','Difference / range, pp':' to '.join(f'{100*x:.2f}' for x in R['missing']['participant_weighted_bounds']),'Interpretation':'Same participant extrema already considered in original supplement'}]
    pd.DataFrame(s).to_csv(TAB/'S8_sensitivity_overview.csv',index=False);md_table(pd.DataFrame(s),TAB/'S8_sensitivity_overview.md')
    for fname,source in [('S6_missingness_descriptors','missingness_descriptors'),('S7_hospital_case_mix','hospital_case_mix')]:
        source_rows=json.loads((OUT/(source+'.json')).read_text());lines=[]
        for row in source_rows:
            ids={k:v for k,v in row.items() if not isinstance(v,dict)}
            for var,val in row.items():
                if not isinstance(val,dict):continue
                if 'levels' in val:
                    for level,n in val['levels'].items():lines.append({**ids,'variable':var,'summary':'category '+level,'value':n})
                else:
                    for stat,v in val.items():lines.append({**ids,'variable':var,'summary':stat,'value':v})
        pd.DataFrame(lines).to_csv(TAB/(fname+'.csv'),index=False)

def plots():
    fig,ax=plt.subplots(figsize=(7.2,3.6));fig.subplots_adjust(left=.06,right=.94,top=.98,bottom=.04)
    ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    ax.text(.5,.91,'RELEASED COHORT',ha='center',va='center',fontsize=8.5,weight='bold',color=MUTED)
    ax.text(.5,.82,f'{int(H.enrolled.sum()):,} participants in 6 randomized hospitals',ha='center',va='center',fontsize=12,weight='bold')
    ax.plot([.5,.5],[.75,.68],color=INK,lw=.9)
    ax.plot([.25,.75],[.68,.68],color=INK,lw=.9)
    for x,a,label in [(.08,1,'TRAINING'),(.55,2,'CONTROL')]:
        z=H[H.arm.eq(a)];n=int(z.enrolled.sum());o=int(z.observed.sum());de=int(z.deaths.sum());lo=int(z.lost.sum())
        center=x+.185
        ax.plot([center,center],[.68,.63],color=INK,lw=.9)
        ax.text(x,.57,label,ha='left',va='center',fontsize=9.5,weight='bold')
        ax.text(x+.37,.57,'3 hospitals',ha='right',va='center',fontsize=8.5,color=MUTED)
        ax.plot([x,x+.37],[.52,.52],color=INK,lw=.9)
        for y,k,v in zip([.43,.34,.25,.16,.07],
                         ['Enrolled','Vital status recorded','Survived','Died','Recorded lost'],
                         [n,o,o-de,de,lo]):
            ax.text(x,y,k,va='center',fontsize=9.5)
            ax.text(x+.37,y,f'{v:,}',ha='right',va='center',fontsize=9.5)
    save(fig,'figure1_flow',['tables/arm_counts.csv','tables/table1_hospitals.csv'])
    fig=plt.figure(figsize=(7.2,3.7))
    ax=fig.add_axes([.15,.18,.52,.77]);tab=fig.add_axes([.70,.18,.28,.77])
    rows=list(H.itertuples());ys=[6.35,5.35,4.35,2.65,1.65,.65]
    for r,y in zip(rows,ys):
        risk=100*r.risk
        ax.hlines(y,0,risk,color=RULE,lw=1.2)
        ax.scatter([risk],[y],s=42,facecolors=INK if r.arm==1 else 'white',edgecolors=INK,linewidths=1.2,zorder=3)
        tab.text(.03,y,f'{int(r.deaths)}/{r.observed}',va='center',fontsize=9)
        tab.text(.56,y,f'{r.lost}/{r.enrolled}',va='center',fontsize=9)
    for a,y,label in [(1,7.45,'TRAINING'),(2,3.75,'CONTROL')]:
        mean=100*H.loc[H.arm.eq(a),'risk'].mean()
        ax.text(0,y,f'{label}   Mean {mean:.2f}%',fontsize=9.2,weight='bold',va='center')
    ax.axhline(3.97,color=LIGHT,lw=.8);tab.axhline(3.97,color=LIGHT,lw=.8)
    ax.set(xlim=(-.8,22),ylim=(.1,8),xticks=[0,5,10,15,20],xlabel='Recorded mortality (%)')
    ax.set_yticks(ys,[f'H{r.hospital}' for r in rows]);ax.tick_params(axis='y',length=0,pad=7)
    ax.spines['left'].set_visible(False);ax.grid(axis='x',color=LIGHT,lw=.6)
    tab.set(xlim=(0,1),ylim=(.1,8));tab.axis('off')
    tab.text(.03,7.45,'DEATHS / KNOWN',va='center',fontsize=7.7,weight='bold',color=MUTED)
    tab.text(.56,7.45,'LOST / ENROLLED',va='center',fontsize=7.7,weight='bold',color=MUTED)
    save(fig,'figure2_hospital_risks',['tables/table1_hospitals.csv'])
    dist=pd.read_csv(OUT/'randomization_distribution.csv').sort_values('statistic').reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(7.2,2.45));fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.30)
    xx=100*dist.statistic.to_numpy();tail=dist.in_two_sided_tail.to_numpy();obs=100*R['primary']['estimate']
    ax.hlines(.30,-10,10,color=RULE,lw=.8)
    for x,is_tail in zip(xx,tail):
        ax.vlines(x,.30,.82,color=INK if is_tail else RULE,lw=2.2 if is_tail else 1.4)
    ax.vlines(obs,.30,1.03,color=INK,lw=3)
    ax.text(obs+.24,1.12,f'Observed {obs:.2f} pp',fontsize=8.5,ha='left',va='center')
    ax.text(9.8,1.12,f'{int(tail.sum())} of 20 as extreme   P = {R["exact"]["p_value"]:.2f}',fontsize=9,weight='bold',ha='right',va='center')
    ax.set(xlim=(-10.3,10.3),ylim=(0,1.35),yticks=[],xticks=[-10,-5,0,5,10],xlabel='Training minus control (percentage points)')
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(axis='x',length=3,color=INK)
    save(fig,'figure3_randomization',['outputs/motor/randomization_distribution.csv'])
    surface=pd.read_csv(OUT/'missing_risk_surface.csv');z=100*surface.pivot(index='q_control',columns='q_training',values='equal_hospital_difference').to_numpy()
    fig,ax=plt.subplots(figsize=(7.2,4.55));fig.subplots_adjust(left=.15,right=.97,top=.96,bottom=.17)
    q=np.linspace(0,100,101)
    ax.contourf(q,q,z,levels=[-100,0,100],colors=[SHADE,'white'])
    ax.contour(q,q,z,levels=[-5],colors=[MUTED],linestyles=['--'],linewidths=[1.1])
    ax.contour(q,q,z,levels=[0],colors=[INK],linewidths=[1.7])
    ax.plot([0,100],[0,100],color=RULE,ls=':',lw=1.2)
    ax.text(23,73,'Training hospitals\nhave lower completed mortality',fontsize=9.5,ha='center',va='center')
    ax.text(54,23,'-5 pp',rotation=32,ha='center',va='center',fontsize=8.5,
            bbox={'facecolor':SHADE,'edgecolor':'none','pad':1})
    ax.text(86,9,'0 pp',rotation=32,ha='center',va='center',fontsize=8.5,
            bbox={'facecolor':'white','edgecolor':'none','pad':1})
    ax.text(62,68,'Equal assumed risk',rotation=45,fontsize=8,color=MUTED,ha='center',va='center')
    ax.set(xlim=(0,100),ylim=(0,100),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100],
           xlabel='Assumed death risk among training losses (%)',
           ylabel='Assumed death risk among control losses (%)')
    ax.tick_params(length=3)
    save(fig,'figure4_missing_outcomes',['outputs/motor/missing_risk_surface.csv'])
    f=pd.read_csv(OUT/'missing_integer_frontier.csv');fig,ax=plt.subplots(figsize=(7.2,3.8))
    fig.subplots_adjust(left=.12,right=.96,top=.96,bottom=.18)
    xx=[];pos=[];guar=[]
    for kc,z in f.groupby('control_unknown_deaths'):
        a=z[z.sign_reversal_possible];b=z[z.sign_reversal_all_distributions]
        xx.append(kc);pos.append(a.training_unknown_deaths.min() if len(a) else np.nan);guar.append(b.training_unknown_deaths.min() if len(b) else np.nan)
    ax.step(xx,pos,where='mid',color=INK,label='Sign reversal possible for some hospital distributions');ax.step(xx,guar,where='mid',color=MUTED,ls='--',label='Sign reversal for every hospital distribution')
    ax.set(xlim=(-.2,15.2),ylim=(29,45),xticks=[0,2,4,6,8,10,12,14],yticks=[30,34,38,42,44],
           xlabel='Unobserved deaths among control losses',ylabel='Minimum deaths among training losses')
    ax.axhline(44,color=RULE,lw=.8);ax.legend(loc='lower right',fontsize=8,frameon=False)
    save(fig,'figureS1_integer_frontier',['outputs/motor/missing_integer_frontier.csv'])
if __name__=='__main__':
    tables();plots();(OUT/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
