"""Compare descriptive PSTH response widths with stored spike waveforms."""
import json
import zipfile

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.analysis import sha256, write_json
from src.experiments.responses.region_peak_timing import fdr_bh
from src.data.io import resolve
from src.data.recording import Recording
from src.data.task import recording_path, source_fingerprint
from src.common.plotting import STYLE


def run(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    source, out = resolve(cfg.source), resolve(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(source / 'unit_selection.csv')
    info = json.loads((source / 'method.json').read_text())
    recording = Recording(recording_path(cfg))
    if info.get('recording_source') != source_fingerprint(recording.path):
        raise ValueError('Response widths do not match the selected Brainsets recording')
    raw = recording.units
    rows = units.source_unit_row.to_numpy(int)
    assert np.array_equal(raw.iloc[rows].id.to_numpy(), units.id)
    attributes = {}
    for name in ['peak_to_valley', 'half_width', 'peak_trough_ratio', 'amplitude', 'snr']:
        units['waveform_' + name] = raw.iloc[rows][name].to_numpy(float)
        attributes[name] = dict(recording.file['units'][name].attrs)
    units['waveform_duration_ms'] = units.waveform_peak_to_valley * 1000
    units['waveform_half_width_ms'] = units.waveform_half_width * 1000
    units['waveform_peak_trough_magnitude_ratio'] = units.waveform_peak_trough_ratio.abs()
    units['response_width_ms'] = units.peak_width_s * 1000
    # Invalid/nonpositive responses have no broad/short response label.
    units['eligible'] = units.peak_width_s.notna() & units.curve_peak_increase_hz.gt(0)
    units['broad_response'] = units.eligible & units.peak_width_s.gt(.3 + 1e-12)
    assert units.loc[units.eligible & ~units.broad_response, 'selection_included'].all()

    waves = []
    for i, unit in units.iterrows():
        waveform = recording.waveform(int(unit.source_unit_row))
        channels = recording.waveform_channels(int(unit.source_unit_row))
        if 'channel' not in channels:
            raise ValueError('Canonical waveform channels lack numeric channel identities')
        channel = np.flatnonzero(channels.channel.to_numpy() == int(unit.peak_channel))
        if len(channel) != 1 or waveform.shape[1] != len(channels):
            raise ValueError('Peak-channel identity does not match canonical waveform channel metadata')
        waves.append(np.asarray(waveform[:, int(channel[0])], float))
        if i % 100 == 0:
            print(f'Read peak-channel waveforms: {i+1}/{len(units)}', flush=True)
    recording.close()
    waves = np.array(waves)
    assert np.isfinite(waves).all()
    centered = waves - waves[:, :10].mean(axis=1, keepdims=True)
    scales = np.abs(centered).max(axis=1, keepdims=True)
    assert np.all(scales > 0)
    normalized = centered / scales
    units['waveform_trough_sample'] = centered.argmin(axis=1)
    # Align by the spike trough and use only samples present in every unit.
    troughs = units.waveform_trough_sample.to_numpy(int)
    before, after = int(troughs.min()), int((centered.shape[1] - 1 - troughs).min())
    offsets = np.arange(-before, after + 1)
    aligned = normalized[np.arange(len(units))[:, None], troughs[:, None] + offsets]
    assert aligned.shape[1] >= int(cfg.waveform_pcs)
    np.savez_compressed(out / 'waveforms.npz', global_unit_id=units.global_unit_id.to_numpy(str),
        raw_peak_channel_waveforms=waves, aligned_normalized_waveforms=aligned,
        sample_offset_from_trough=offsets)

    features = ['waveform_duration_ms', 'waveform_half_width_ms', 'waveform_peak_trough_magnitude_ratio']
    scopes = {'All units': units.eligible, 'CP': units.eligible & units.region.eq('CP'),
        'ACB': units.eligible & units.region.eq('ACB'),
        'Single units': units.eligible & units.decoder_label.eq('sua'),
        'CP single units': units.eligible & units.region.eq('CP') & units.decoder_label.eq('sua'),
        'ACB single units': units.eligible & units.region.eq('ACB') & units.decoder_label.eq('sua')}
    associations = []
    for scope, mask in scopes.items():
        group = units.loc[mask]
        for feature in features:
            short = group.loc[~group.broad_response, feature].to_numpy(float)
            broad = group.loc[group.broad_response, feature].to_numpy(float)
            assert np.isfinite(np.r_[short, broad]).all()
            rho, p_rho = spearmanr(group[feature], group.response_width_ms)
            u, p_u = mannwhitneyu(broad, short, alternative='two-sided')
            associations.append(dict(scope=scope, feature=feature, n=len(group),
                n_short=len(short), n_broad=len(broad), rho=rho, spearman_p=p_rho,
                short_median=np.median(short), broad_median=np.median(broad),
                short_q25=np.quantile(short, .25), short_q75=np.quantile(short, .75),
                broad_q25=np.quantile(broad, .25), broad_q75=np.quantile(broad, .75),
                probability_broad_has_larger_value=u/(len(short)*len(broad)), mannwhitney_p=p_u))
    associations = pd.DataFrame(associations)
    associations['spearman_q'] = fdr_bh(associations.spearman_p.to_numpy())
    associations['mannwhitney_q'] = fdr_bh(associations.mannwhitney_p.to_numpy())
    associations.to_csv(out / 'associations.csv', index=False)

    predictions, prediction_summaries = [], []
    for scope, mask in scopes.items():
        group = units.loc[mask]
        y = group.broad_response.to_numpy(int)
        # Balance each fold by response class and region when both regions are present.
        strata = group.region.astype(str) + '_' + group.broad_response.astype(str)
        folds = min(int(cfg.folds), int(strata.value_counts().min()))
        assert folds >= 2
        for model in ['Waveform metrics', 'Waveform shape (PCA)']:
            x = (group[features].to_numpy(float) if model == 'Waveform metrics'
                 else aligned[group.index])
            scores = []
            for repeat in range(int(cfg.repeats)):
                splitter = StratifiedKFold(folds, shuffle=True, random_state=int(cfg.seed)+repeat)
                prob = np.full(len(group), np.nan)
                for train, test in splitter.split(x, strata):
                    classifier = LogisticRegression(C=1, class_weight='balanced', max_iter=2000)
                    if model == 'Waveform metrics':
                        estimator = make_pipeline(StandardScaler(), classifier)
                    else:
                        estimator = make_pipeline(PCA(n_components=min(int(cfg.waveform_pcs), len(train)-1),
                            svd_solver='full'), StandardScaler(), classifier)
                    estimator.fit(x[train], y[train])
                    prob[test] = estimator.predict_proba(x[test])[:, 1]
                assert np.isfinite(prob).all()
                scores.append(dict(roc_auc=roc_auc_score(y, prob),
                    balanced_accuracy=balanced_accuracy_score(y, prob >= .5),
                    average_precision=average_precision_score(y, prob)))
                predictions.extend(dict(scope=scope, model=model, repeat=repeat, global_unit_id=identifier,
                    broad_response=int(label), heldout_probability=float(p))
                    for identifier, label, p in zip(group.global_unit_id, y, prob))
            score_frame = pd.DataFrame(scores)
            record = dict(scope=scope, model=model, n=len(group), n_short=int((y == 0).sum()),
                          n_broad=int(y.sum()), folds=folds, repeats=int(cfg.repeats), broad_prevalence=y.mean())
            for metric in score_frame:
                record[metric + '_mean'] = score_frame[metric].mean()
                record[metric + '_min'] = score_frame[metric].min()
                record[metric + '_max'] = score_frame[metric].max()
            prediction_summaries.append(record)
    prediction_summaries = pd.DataFrame(prediction_summaries)
    prediction_summaries.to_csv(out / 'prediction_summary.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'heldout_predictions.csv', index=False)
    units.to_csv(out / 'units.csv', index=False)

    colors = {False: '#94A3B8', True: '#E07832'}
    labels = ['Spike duration (ms)', 'Spike half-width (ms)']
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True, layout='constrained')
        for row, region in enumerate(['CP', 'ACB']):
            group = units.loc[scopes[region]]
            for col, (feature, label) in enumerate(zip(features[:2], labels)):
                ax = axes[row, col]
                for broad in [False, True]:
                    subset = group.loc[group.broad_response.eq(broad)]
                    ax.scatter(subset[feature], subset.response_width_ms, s=23,
                               color=colors[broad], alpha=.7, edgecolors='white', linewidths=.35)
                ax.axhline(300, color='#64748B', lw=.9, ls='--')
                ax.set_yscale('log')
                ax.set(xlabel=label, title=f'{region} · {len(group)} units')
                ax.set_yticks([50, 100, 300, 1000, 3000], ['50', '100', '300', '1,000', '3,000'])
                ax.set_ylim(25, 3000)
                ax.grid(axis='y', alpha=.12)
                ax.set_axisbelow(True)
                if col == 0:
                    ax.set_ylabel('Response width (ms)')
        for col, feature in enumerate(features[:2]):
            maximum = units.loc[units.eligible, feature].max()
            for row in range(2):
                axes[row, col].set_xlim(0, maximum * 1.05)
        fig.legend(handles=[Line2D([], [], marker='o', ls='', color=colors[b],
            label='Broad response (>300 ms)' if b else 'Short response (≤300 ms)') for b in [False, True]],
            loc='outside upper center', ncol=2)
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'waveform_response_width.{ext}', dpi=240, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True, layout='constrained')
        for ax, region in zip(axes, ['CP', 'ACB']):
            for broad in [False, True]:
                indices = units.index[scopes[region] & units.broad_response.eq(broad)]
                subset = aligned[indices]
                median = np.median(subset, axis=0)
                lo, hi = np.quantile(subset, [.25, .75], axis=0)
                ax.fill_between(offsets, lo, hi, color=colors[broad], alpha=.18)
                ax.plot(offsets, median, color=colors[broad], lw=2,
                        label=f'{"Broad" if broad else "Short"} · {len(indices)} units')
            ax.set(title=region, xlabel='Sample offset from spike trough')
            ax.axhline(0, color='#CBD5E1', lw=.8)
            ax.legend()
        axes[0].set_ylabel('Normalized spike amplitude')
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'waveform_group_shapes.{ext}', dpi=240, bbox_inches='tight')
        plt.close(fig)

    method = dict(
        recording_id=info['recording_id'], recording_source=info['recording_source'],
        question='Are broad (>300 ms) and short (<=300 ms) trial-averaged PSTH responses associated with spike waveforms in this Dec 10 recording?',
        population='Start with all 318 CP/ACB non-noise units, retaining SUA and MUA. Analyze only positive peaks with fully measured response width. Flat/nonpositive/width-censored units have no class label. Repeat within regions and SUA subsets.',
        response='Reuse the existing 20-ms PSTH Gaussian-smoothed (sigma 20 ms) baseline-relative half-height peak width. This does not measure complete sustained tails. No response widths are recomputed here.',
        waveform='Use preserved Brainsets peak-to-valley duration, half-width (converted seconds to ms), and magnitude of peak/trough ratio. Also read the full mean waveform on the identified peak channel, subtract the first-ten-sample mean, normalize by maximum absolute excursion, align to trough, and retain the common sample support. Amplitude and SNR are saved but excluded from shape prediction. Sampling frequency of the stored waveform array is not assumed; shape plots use sample index.',
        associations='Spearman correlations with continuous response width and two-sided Mann-Whitney broad/short comparisons. BH-adjust all 18 Spearman tests together and all 18 Mann-Whitney tests together. These are descriptive neuron-level associations within one recording; shared population dependence is not addressed by the nominal p-values.',
        prediction='Five repeated five-fold stratified unit holdouts, stratifying by broad/short and region; reduce fold count only if a stratum is too small. Fixed logistic regression C=1 and balanced class weights. First model uses three waveform metrics; second uses ten PCs of normalized aligned waveform samples. PCA and scaling are fitted on training units only. No tuning, no firing rate, region, cell labels, or PSTH data as predictors. Repeated-split score range is not a confidence interval. AUC 0.5 means chance ranking; average-precision baseline equals broad-response prevalence. All held-out units are from the same session.',
        plots='Scatter panels show every eligible unit; Y uses a logarithmic response-width axis, and the dashed line is 300 ms. Waveform group curves are medians and IQRs, not confidence intervals.',
        waveform_source_attributes=attributes,
        source_hashes={str(p):sha256(p) for p in [resolve(__file__), source / 'unit_selection.csv', source / 'method.json']})
    write_json(out / 'method.json', method)
    OmegaConf.save(cfg, out / 'config.yaml')
    summary = dict(input_units=len(units), eligible_units=int(units.eligible.sum()),
        broad_units=int(units.broad_response.sum()), short_units=int((units.eligible & ~units.broad_response).sum()),
        raw_waveform_shape=list(waves.shape), aligned_waveform_shape=list(aligned.shape),
        trough_sample_min=int(troughs.min()), trough_sample_max=int(troughs.max()),
        associations=associations.to_dict(orient='records'), prediction=prediction_summaries.to_dict(orient='records'))
    write_json(out / 'summary.json', summary)
    (out / 'index.html').write_text('\n'.join([
        '<!doctype html><meta charset="utf-8"><title>Response width and spike waveform</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#243247}img{max-width:100%}td,th{padding:6px}</style>',
        '<h1>Response width and spike waveform · December 10</h1>',
        '<p>Broad/short refers to the duration of the trial-averaged response, not spike duration. Waveform comparisons use independently stored raw NWB measurements.</p>',
        '<img src="waveform_response_width.png"><img src="waveform_group_shapes.png">',
        prediction_summaries[['scope','model','n','n_broad','roc_auc_mean','balanced_accuracy_mean']].to_html(index=False),
        '<p><a href="units.csv">Matched units</a> · <a href="associations.csv">Associations</a> · <a href="prediction_summary.csv">Held-out prediction</a> · <a href="method.json">Methods</a></p>']) + '\n')
    with zipfile.ZipFile(out / 'figures_and_tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.suffix in ['.png','.pdf','.csv','.json','.yaml','.html','.npz']:
                archive.write(path, path.name)
    print(associations[['scope','feature','n_short','n_broad','rho','spearman_q','short_median','broad_median','mannwhitney_q']].to_string(index=False), flush=True)
    print(prediction_summaries[['scope','model','n','n_broad','roc_auc_mean','roc_auc_min','roc_auc_max','balanced_accuracy_mean']].to_string(index=False), flush=True)
    print('Completed:', out, flush=True)
