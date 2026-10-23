import numpy as np
from src.common.plotting import STYLE


def plot(results, neurons, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ['Probability', 'Choice', 'Outcome', 'Licking']
    maximum = float(results.loc[results.group.isin(groups), 'partial_r2'].max() * 100)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        fig.subplots_adjust(left=.10, right=.89, bottom=.21, top=.78, wspace=.34)
        for ax, mode in zip(axes, ['Raw', 'Adjusted']):
            subset = results.loc[results['mode'].eq(mode)].set_index(['neuron', 'group'])
            values = np.array([[subset.loc[(neuron, group), 'partial_r2'] * 100 for group in groups] for neuron in neurons])
            image = ax.imshow(values, cmap='Blues', vmin=0, vmax=maximum, aspect='auto')
            ax.set_xticks(np.arange(len(groups)), groups, fontsize=10)
            ax.set_yticks(np.arange(len(neurons)), neurons, fontsize=10)
            ax.tick_params(length=0, pad=7)
            for i, neuron in enumerate(neurons):
                for j, group in enumerate(groups):
                    row = subset.loc[(neuron, group)]
                    mark = '*' if row.q < .05 else ''
                    ax.text(j, i, f'{values[i,j]:.1f}{mark}', ha='center', va='center',
                            color='white' if values[i,j] > maximum * .55 else '#1E293B', fontsize=11)
            for boundary in [1.5, 3.5, 5.5]:
                ax.axhline(boundary, color='white', lw=2)
            ax.set_title('Raw scores' if mode=='Raw' else 'Class-centered scores', pad=15, fontsize=13)
            for spine in ax.spines.values():
                spine.set_visible(False)
        bar = fig.colorbar(image, cax=fig.add_axes([.92, .26, .016, .45]))
        bar.set_label('Unique residual variance explained (%)', labelpad=10)
        bar.outline.set_visible(False)
        fig.suptitle('Task associations beyond session trends', x=.10, y=.96, ha='left', fontsize=17)
        fig.text(.10, .89, '745502 · Dec 10 · Sep 12 model · Selected pairs', color='#64748B', fontsize=11)
        fig.text(.10, .12, 'Each task group is tested conditional on the others, time trends, movement and availability.', fontsize=9)
        fig.text(.10, .075, '* Exploratory q < 0.05 · 5-minute spline knots · 60-second HAC lag · 80 tests', fontsize=9)
        fig.text(.10, .03, 'Values describe in-sample effects; pair selection used this same recording.', color='#64748B', fontsize=9)
        for suffix in ['png','pdf']:
            fig.savefig(output / f'task_associations.{suffix}', dpi=240, bbox_inches='tight', pad_inches=.18)
        plt.close(fig)
