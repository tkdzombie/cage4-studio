import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);p.add_argument('--initial-label',default='Fresh');p.add_argument('--title',default='CAGE4 / flat PPO from primitive actions');a=p.parse_args()
rows=[json.loads(x) for x in (Path(a.run)/'metrics.jsonl').read_text().splitlines()]
eval_data=json.loads(Path(a.evaluation).read_text())
x=np.array([r['env_steps'] for r in rows]);y=np.array([r['return'] for r in rows])
fig,axs=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
axs[0].plot(x,y,color='#93acc3',lw=1,marker='.',label='Raw update mean')
if len(y)>=5:axs[0].plot(x[4:],np.convolve(y,np.ones(5)/5,mode='valid'),color='#174e78',lw=2,label='5-update mean')
axs[0].set(xlabel='Environment transitions',ylabel='Raw team return (higher is better)',title='One training seed; full 499-step episodes')
axs[0].legend(frameon=False,fontsize=8)
labels=[k for k in ['initial','latest','final','author-restricted','author'] if k in eval_data['means']]
seeds=sorted({r['seed'] for r in eval_data['rows']})
for i,label in enumerate(labels):
    scores=[r['return'] for r in eval_data['rows'] if r['endpoint']==label]
    axs[1].scatter(np.linspace(i-.08,i+.08,len(scores)),scores,color='#9ba6b0',s=25,zorder=2)
    axs[1].plot([i-.2,i+.2],[np.mean(scores)]*2,color='#174e78' if not label.startswith('author') else '#94703d',lw=3,zorder=3)
    axs[1].annotate(f'{np.mean(scores):.1f}',(i,np.mean(scores)),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
axs[1].set(xticks=range(len(labels)),xticklabels=[{'initial':a.initial_label,'latest':'Intermediate','final':'Final','author':'Author weights','author-restricted':'Author (no traffic)'}[k] for k in labels],ylabel='Raw team return',title=f'Paired scenarios; shared exploration seeds (n={len(seeds)})')
for ax in axs:
    ax.grid(axis='y',color='#e5e8eb',lw=.6);ax.spines[['top','right']].set_visible(False)
fig.suptitle(a.title,fontweight='bold')
fig.savefig(a.output,dpi=170)
