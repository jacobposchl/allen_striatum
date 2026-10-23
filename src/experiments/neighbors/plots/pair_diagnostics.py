import numpy as np
from src.common.plotting import STYLE


def plot(window, units, controls, splits, waves, isi_edges, isi_counts, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#7451AA','#179B83']
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,3,figsize=(16,9))
        fig.subplots_adjust(left=.075,right=.97,top=.85,bottom=.11,wspace=.35,hspace=.60)
        time=(window.start_session_s+5)/60
        for s,unit in enumerate([156,290]):
            axes[0,0].plot(time,window[f'rate_{unit}'],lw=.8,color=colors[s],label=f'D2 {unit}')
        axes[0,0].set(title=f'Firing rates · r = {summary["raw_rate_r"]:.2f}',xlabel='Session time (min)',ylabel='Spikes / s')
        axes[0,0].legend(fontsize=9)
        axes[0,1].hist(window.shared_neighbors,bins=np.arange(-.5,11,1),color='#66889F',rwidth=.85)
        axes[0,1].set(title='Shared nearest neighbors',xlabel='Shared neighbors (of 10)',ylabel='Windows',xticks=[0,2,4,6,8,10])
        positions=[0,1]
        for position,(kind,color) in enumerate([('Shared reference half','#66889F'),('Disjoint reference halves','#C98A54')]):
            values=splits.loc[splits.reference.eq(kind),'r'].to_numpy()
            axes[0,2].boxplot(values,positions=[position],widths=.5,patch_artist=True,
                              boxprops=dict(facecolor=color,alpha=.7),medianprops=dict(color='#1E293B'),showfliers=False)
        axes[0,2].axhline(summary['raw_score_r'],color='#475569',ls='--',lw=1,label='Original r')
        axes[0,2].set(title='Independent reference pools',ylabel='Score correlation',xticks=positions,xticklabels=['Same half','Disjoint halves'])
        axes[0,2].legend(fontsize=8)
        selected=controls.loc[controls['mode'].eq('Raw')].copy()
        # Pandas treats the saved literal "None" as missing when rereading CSV.
        selected['controls']=selected['controls'].fillna('None')
        names=['None','Both rates (nonlinear)','Time + controls + task','Time + controls + task + both rates']
        values=[float(selected.loc[selected.controls.eq(name),'r'].iloc[0]) for name in names]
        axes[1,0].barh(range(4),values,color=['#94A3B8','#9A83BA','#76ADA5','#326C79'])
        axes[1,0].set(yticks=range(4),yticklabels=['Original','Rates','Time + task + controls','All + rates'],xlabel='Residual score correlation',title='Accounting for firing rates',xlim=(min(0,min(values)-.1),max(.8,max(values)+.1)))
        axes[1,0].invert_yaxis()
        for y,value in enumerate(values): axes[1,0].text(value+.015,y,f'{value:.2f}',va='center',fontsize=9)
        for s,unit in enumerate([156,290]):
            axes[1,1].plot(np.arange(waves.shape[1]),waves[s],color=colors[s],lw=1.5,label=f'D2 {unit}')
            widths=np.diff(isi_edges)*1000
            axes[1,2].stairs(isi_counts[s]/isi_counts[s].sum()/widths,isi_edges*1000,color=colors[s],lw=1.3,baseline=None)
        axes[1,1].set(title='Own peak-channel waveforms',xlabel='Stored waveform sample',ylabel='Normalized amplitude')
        axes[1,1].legend(fontsize=9)
        axes[1,2].set(xscale='log',title='Interspike intervals',xlabel='ISI (ms)',ylabel='Density / ms')
        for ax in axes.flat:
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('D2 156 & D2 290 · Pair diagnostics',x=.075,y=.97,ha='left',fontsize=18)
        fig.text(.075,.91,f"Both MUA · {summary['brain_regions'][0]} / {summary['brain_regions'][1]} · CCF separation {summary['ccf_distance_um']:.0f} µm",fontsize=12,color='#64748B')
        fig.text(.075,.025,'Reference-pool boxes: 100 fixed splits × 2 orientations, not confidence intervals. Rate-control bars: 425 matched windows.',fontsize=10,color='#64748B')
        for suffix in ['png','pdf']: fig.savefig(output/f'pair_diagnostics.{suffix}',dpi=230,bbox_inches='tight',pad_inches=.2)
        plt.close(fig)
