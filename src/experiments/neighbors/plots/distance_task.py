import numpy as np
from src.common.plotting import STYLE


def plot(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main=results.loc[results.variant.eq('Primary')].copy().reset_index(drop=True)
    colors={'Probability':'#3078B5','Choice':'#8A65AD','Outcome':'#C9853E','Licking':'#248B7D'}
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(12,max(6,len(main)*.42+2)))
        fig.subplots_adjust(left=.30,right=.95,bottom=.18,top=.82)
        for y,row in main.iterrows():
            ax.barh(y,row.heldout_gain*100,color=colors[row.group],alpha=.85)
            if row.q<.05 and row.heldout_gain>0:ax.text(row.heldout_gain*100,y,' *',va='center')
        ax.set_yticks(np.arange(len(main)),[f'{r.group} · {r.label_a} {int(r.unit_a)} / {r.label_b} {int(r.unit_b)}' for r in main.itertuples()],fontsize=9)
        ax.invert_yaxis();ax.axvline(0,color='#475569',lw=1)
        ax.set_xlabel('Held-out error reduction from task group (%)')
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
        fig.suptitle('Direct embedding distance · Task associations',x=.05,y=.97,ha='left',fontsize=17)
        fig.text(.05,.90,'Dec 10 · Sep 12 model · Candidates selected on training windows only',fontsize=11,color='#64748B')
        fig.text(.05,.09,'Positive: task improves held-out predictions. Negative: task worsens them.',fontsize=10)
        fig.text(.05,.04,'Controls: time, firing rates, population distance, movement and availability. * Validation q < 0.05.',fontsize=9,color='#64748B')
        for suffix in ['png','pdf']:fig.savefig(output/f'heldout_task_associations.{suffix}',dpi=230,bbox_inches='tight',pad_inches=.18)
        plt.close(fig)
