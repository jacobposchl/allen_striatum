"""Sequential training within an existing allocation; never submits jobs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

from src.cli import load_config


ROOT = Path(__file__).resolve().parents[2]


def run(cfg):
    output = Path(cfg.output_root)
    if not output.is_absolute():
        output = ROOT / output
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output / f"job_{os.environ.get('SLURM_JOB_ID', 'manual')}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    state = {"status": "running", "job_id": os.environ.get("SLURM_JOB_ID"),
             "output": str(output), "stages": {}, "checkpoints": {}, "embeddings": {}}

    def persist():
        temporary = output / "pipeline.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "pipeline.json")

    def execute(name, command, settings):
        stage = state["stages"][name] = {"status": "running"}
        config_name = f"{name}.yaml"
        config_path = output / config_name
        OmegaConf.save(settings, config_path, resolve=True)
        # Each subprocess releases all GPU memory before the next stage starts.
        stage["config"] = str(config_path)
        persist()
        print(f"Starting {name}; results: {output}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-u", "-m", "src.training.pipeline",
                 command, str(config_path)], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, check=True,
            )
        stage["status"] = "completed"
        persist()

    def train(name, settings):
        settings.output_root = str(output / name)
        settings.export.enabled = False
        execute(name, name, settings)
        summaries = list((output / name).glob("*/*/*/summary.json"))
        if len(summaries) != 1:
            raise RuntimeError(f"Expected exactly one completed {name} run, found {len(summaries)}")
        summary = json.loads(summaries[0].read_text())
        if summary["completed_epochs"] != settings.epochs:
            raise RuntimeError(f"{name} did not complete its configured epochs")
        checkpoint = Path(summary["selected_checkpoint"]).resolve(strict=True)
        state["checkpoints"][name] = str(checkpoint)
        persist()
        return str(checkpoint)

    persist()
    try:
        pretrain = load_config("pretrain", [], cfg.pretrain_config)
        finetune = load_config("finetune", [], cfg.finetune_config)
        passive_checkpoint = train("pretrain", pretrain)
        finetune.checkpoint = passive_checkpoint
        active_checkpoint = train("finetune", finetune)
        # Preserve both encoder spaces on both cohorts for before/after readouts.
        # Export adapted active data first, the immediate analysis target.
        for encoder, checkpoint in (("finetune", active_checkpoint), ("pretrain", passive_checkpoint)):
            for dataset, settings in (("active", finetune), ("passive", pretrain)):
                name = f"export_{encoder}_{dataset}"
                export = OmegaConf.create({
                    "data": OmegaConf.to_container(settings.data, resolve=True),
                    "cohort": OmegaConf.to_container(settings.cohort, resolve=True),
                    "checkpoint": checkpoint, "output": str(output / "embeddings" / encoder / dataset),
                    "device": "cuda", "batch_size": cfg.export.batch_size,
                    "save_latents": cfg.export.save_latents, "max_windows": None,
                    "cpu_threads": cfg.cpu_threads,
                })
                execute(name, "export", export)
                state["embeddings"][f"{encoder}_{dataset}"] = export.output
                persist()
        state["status"] = "completed"
        persist()
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = str(error)
        for stage in state["stages"].values():
            if stage["status"] == "running":
                stage["status"] = "failed"
        persist()
        raise
    print(f"Pipeline completed: {output / 'pipeline.json'}", flush=True)
    return state


if __name__ == "__main__":
    # Use fully resolved saved settings without reconstructing Hydra defaults.
    from src.cli import dispatch, require_compute
    command, config_path = sys.argv[1:]
    settings = OmegaConf.load(config_path)
    require_compute(command, settings)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(int(settings.get("cpu_threads", 1)))
    dispatch(command, settings)
