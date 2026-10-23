"""Keep command discovery and configuration inspection independent of numerical work."""
import argparse
import os
import socket
from pathlib import Path


COMMANDS = {
    "pretrain": "Pretrain NuCLR on the selected cohort",
    "finetune": "Adapt pretrained weights to a new cohort",
    "training_pipeline": "Pretrain, fine-tune, and export from Brainsets HDF5 recordings",
    "celltypes": "Fit/evaluate a cell-type readout or predict probabilities",
    "export": "Export embeddings from a checkpoint",
    "glm_pilot": "Fit rich behavioral GLMs with blocked validation and full response curves",
    "glm_cache": "Cache full-recording GLMs and separate validation across active recordings",
    "probe": "Compare linear decoding of session and animal identity",
    "neighbors": "Plot labeled-neighbor agreement across embedding windows",
    "neighbor_pairs": "Compare raw and class-centered neighbor-score correlations",
    "pair_traces": "Plot selected neighbor-score pairs with aligned task records",
    "task_associations": "Test score associations with task variables beyond session trends",
    "pair_diagnostics": "Check shared neighbors, firing rates and unit features for a pair",
    "distance_task": "Discover and validate task associations of direct embedding distances",
    "cluster": "Cluster each recording's embeddings and compare electrophysiology features",
    "cluster_glm": "Compare saved recording clusters with rich GLM responses",
    "cluster_glm_cache": "Compare recording clusters across the final GLM cache",
    "cluster_task_responses": "Inspect observed and held-out task responses of saved clusters",
    "plot": "View embeddings and saved neuron-level features",
    "joint_umap": "Plot passive and active embeddings from the same encoder",
    "encoder_comparison": "Compare matched units and unsupervised structure across encoders",
    "passive_cluster_enrichment": "Evaluate labeled-cell enrichment in passive unsupervised clusters",
    "encoder_cluster_glm": "Compare encoder clusters with cached functional properties",
    "encoder_shared_prediction": "Test shared encoder axes and held-out GLM-property prediction",
    "passive_transfer": "Match a passive reference group across recordings and inspect behavior",
    "active_choice_transfer": "Describe cluster choice responses and test their transfer across active recordings",
    "active_cue_axes": "Interpret go-cue kernel axes and test frozen axes across recordings",
    "active_feature_transfer": "Screen individual GLM features for bidirectional recording transfer",
    "active_outcome_observed": "Check frozen embedding outcome predictions against observed firing",
    "glm_embedding_maps": "View individual GLM effects on fixed within-session UMAP and PCA coordinates",
    "population_psth": "Plot observed population PSTHs grouped by region and ordered by depth",
    "region_peak_timing": "Compare observed CP and ACB post-cue peak-time distributions",
    "cue_response_glm": "Fit five-basis go-cue response curves and validate across three trial periods",
    "timing_sorted_psth": "View CP and ACB response shapes ordered by peak time with region and cell-tag strips",
    "response_density": "Overlay time-resolved CP and ACB normalized response densities",
    "running_correlation": "View rolling firing/speed correlations at several lags",
    "running_examples": "Compare late-session running and example units with the zero-lag heatmap",
    "running_bouts": "View a run-onset population PSTH and five individual running bouts",
    "movement_licking": "Compute per-unit running-bout and licking profiles with support and temporal checks",
    "trial_event_glm": "Fit five-basis choice/outcome profiles with three-fold blocked prediction checks",
    "movement_licking_review": "Review movement/licking support, sensitivity, and descriptive candidate counts",
}


def load_config(command, overrides, name=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    source = Path(__file__).resolve().parent
    directories = [source / "configs", source / "glm" / "configs",
                   *sorted((source / "experiments").glob("*/configs"))]
    requested = Path(name or command).expanduser()
    if requested.suffix not in {".yaml", ".yml"}:
        requested = requested.with_suffix(".yaml")
    if requested.is_absolute() or requested.parent != Path(".") or requested.is_file():
        candidates = [requested.resolve()] if requested.is_file() else []
    else:
        candidates = [directory / requested for directory in directories
                      if (directory / requested).is_file()]
    if not candidates:
        raise ValueError(
            f"No configuration found for {name or command!r}. "
            "Use --config /path/to/settings.yaml to provide explicit settings."
        )
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous configuration {name or command!r}; use its full path")
    config = candidates[0]
    search = [directory.as_uri() for directory in directories if directory != config.parent]
    with initialize_config_dir(version_base=None, config_dir=str(config.parent)):
        cfg = compose(config_name=config.stem,
                      overrides=[f"hydra.searchpath=[{','.join(search)}]", *overrides])
    OmegaConf.resolve(cfg)
    return cfg


def require_compute(command, cfg):
    """Never start fitting, data loading, or inference on a PARC login node."""
    host = socket.gethostname().split(".")[0].lower()
    saved_plot = command == "plot" and cfg.get("coordinates") and not cfg.get("embeddings")
    if any(word in host for word in ("login", "betty")) and not saved_plot:
        raise RuntimeError(
            f"{command} requires an existing compute allocation; this host is {host}. "
            "Use --show-config here to inspect settings. This program never requests an allocation."
        )


def dispatch(command, cfg):
    if command == "training_pipeline":
        from src.training.pipeline import run
        return run(cfg)
    if command in {"pretrain", "finetune"}:
        from src.training.trainer import NuCLRTrainer
        cfg.stage = command
        return NuCLRTrainer(cfg).train()
    if command == "export":
        from src.embeddings.export import run
    elif command == "celltypes":
        from src.celltypes.workflow import run
    elif command == "glm_pilot":
        from src.glm.pilot import run
    elif command == "glm_cache":
        from src.glm.cache import run
    elif command == "probe":
        from src.experiments.transfer.identity_probe import run
    elif command == "neighbors":
        from src.experiments.neighbors.temporal_neighbors import run
    elif command == "neighbor_pairs":
        from src.experiments.neighbors.neighbor_pairs import run
    elif command == "pair_traces":
        from src.experiments.neighbors.plots.pair_traces import run
    elif command == "task_associations":
        from src.experiments.neighbors.task_associations import run
    elif command == "pair_diagnostics":
        from src.experiments.neighbors.pair_diagnostics import run
    elif command == "distance_task":
        from src.experiments.neighbors.distance_task import run
    elif command == "cluster":
        from src.experiments.clustering.clustering import run
    elif command == "cluster_glm":
        from src.experiments.clustering.cluster_glm import run
    elif command == "cluster_glm_cache":
        from src.experiments.clustering.cluster_glm_cache import run
    elif command == "cluster_task_responses":
        from src.experiments.clustering.cluster_task_responses import run
    elif command == "joint_umap":
        from src.embeddings.joint_umap import run
    elif command == "encoder_comparison":
        from src.experiments.transfer.encoder_comparison import run
    elif command == "passive_cluster_enrichment":
        from src.experiments.clustering.passive_cluster_enrichment import run
    elif command == "encoder_cluster_glm":
        from src.experiments.transfer.encoder_cluster_glm import run
    elif command == "encoder_shared_prediction":
        from src.experiments.transfer.encoder_shared_prediction import run
    elif command == "passive_transfer":
        from src.experiments.transfer.passive_transfer import run
    elif command == "active_choice_transfer":
        from src.experiments.transfer.active_choice_transfer import run
    elif command == "active_cue_axes":
        from src.experiments.transfer.active_cue_axes import run
    elif command == "active_feature_transfer":
        from src.experiments.transfer.active_feature_transfer import run
    elif command == "active_outcome_observed":
        from src.experiments.transfer.active_outcome_observed import run
    elif command == "glm_embedding_maps":
        from src.experiments.responses.glm_embedding_maps import run
    elif command == "population_psth":
        from src.experiments.responses.population_psth import run
    elif command == "region_peak_timing":
        from src.experiments.responses.region_peak_timing import run
    elif command == "cue_response_glm":
        from src.experiments.responses.cue_response import run
    elif command == "timing_sorted_psth":
        from src.experiments.responses.timing_sorted_psth import run
    elif command == "response_density":
        from src.experiments.responses.response_density import run
    elif command == "running_correlation":
        from src.experiments.tuning.running_correlation import run
    elif command == "running_examples":
        from src.experiments.tuning.running_examples import run
    elif command == "running_bouts":
        from src.experiments.tuning.running_bouts import run
    elif command == "movement_licking":
        from src.experiments.tuning.movement_licking import run
    elif command == "trial_event_glm":
        from src.experiments.tuning.trial_event_glm import run
    elif command == "movement_licking_review":
        from src.experiments.tuning.movement_licking_review import run
    elif command == "plot":
        from src.embeddings.plotting import run
    else:
        raise ValueError(f"Unknown workflow: {command}")
    return run(cfg)


def main(argv=None):
    description = "NuCLR striatal analysis. Commands:\n" + "\n".join(
        f"  {name:10s} {description}" for name, description in COMMANDS.items()
    )
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("overrides", nargs="*", metavar="key=value", help="Hydra config overrides, including data=active and cohort=active_745502")
    parser.add_argument("--config", help="Configuration name or path; searches core and analysis-family settings")
    parser.add_argument("--show-config", action="store_true", help="Print resolved configuration without reading data or running analysis")
    args = parser.parse_intermixed_args(argv)
    try:
        cfg = load_config(args.command, args.overrides, args.config)
        if args.show_config:
            from omegaconf import OmegaConf
            print(OmegaConf.to_yaml(cfg, resolve=True))
            return
        require_compute(args.command, cfg)
        # Set limits before numerical libraries are imported. Individual workflows
        # divide their explicit allocation further when they use worker processes.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(max(1, int(cfg.get("cpu_threads", cfg.get("threads", 1)))))
        dispatch(args.command, cfg)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")
