from typing import Any, Optional
import os
import importlib
import math
import yaml

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import pydantic
import hydra
import tqdm
import wandb
import coolname
from hydra.core.hydra_config import HydraConfig

from adam_atan2 import AdamATan2
from arch.layers import Carry

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class TrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data: DataConfig

    # Name used for the wandb group / checkpoint dir (defaults to the hydra config name)
    run_name: Optional[str] = None

    seeds: list[int] = [42]

    cycles_per_data: int
    epochs: int

    local_batch_size: int

    lr: float
    lr_warmup_steps: int
    lr_min_ratio: float

    weight_decay: float
    beta1: float
    beta2: float
    ema: Optional[float] = None

    log_interval: int = 5
    # Evaluate every N optimizer steps instead of once per epoch. Needed when total steps are
    # matched across datasets of very different sizes: on full Sudoku-Extreme one pass over the
    # data is already the whole step budget, so per-epoch eval would yield a single point.
    eval_interval: Optional[int] = None

    # Eval splits, and optionally a different dataset to take them from. Full Sudoku-Extreme names
    # its splits train/test and has no `test_hard`, so the full-data runs point these at the 1k
    # repo to evaluate on exactly the same 20k test_hard puzzles as the 1k runs. Verified disjoint
    # from the full train split (0/20,000 overlap), so this leaks nothing.
    eval_splits: list[str] = ["test_hard"]
    eval_dataset_name: Optional[str] = None
    # Eval split that "best.pt" is selected on (must be one of `eval_splits`)
    best_metric_split: str = "test_hard"

    # Train-to-convergence ("warmup-stable-decay"): after the warm-up the lr stays at `lr` (set
    # lr_min_ratio: 1.0) until the best `best_metric_split` score has not improved by at least
    # `plateau_min_delta` for `plateau_patience_steps` optimizer steps; the lr is then annealed
    # (cosine) to `anneal_min_ratio` x lr over `anneal_steps` steps and training stops. `epochs` only
    # bounds the run. The decision is taken on rank 0 after an eval and broadcast, so all ranks
    # stop together. None (default) = the fixed-length schedule above, no early stop.
    plateau_patience_steps: Optional[int] = None
    plateau_min_delta: float = 0.001
    anneal_steps: int = 83200
    anneal_min_ratio: float = 0.05
    # Early stop for the fixed-length schedule: stop (no anneal) once the best `best_metric_split`
    # score has not improved by `plateau_min_delta` for this many optimizer steps. Meant for tiny
    # training sets that peak early and then overfit (test accuracy falls while train goes to 100 %),
    # where the rest of the budget is wasted; `best.pt` is unaffected. None (default) = run to the end.
    early_stop_patience_steps: Optional[int] = None
    # Fixed run length in optimizer steps, independent of the dataset size: the cosine schedule spans
    # `max_steps` and training stops there (after a final eval). `epochs` must be large enough to reach
    # it. None (default) = the run spans `epochs` and the schedule is `cycles_per_data x batches x epochs`.
    max_steps: Optional[int] = None

    # Warm restart from a saved state_dict (e.g. an earlier run's `last.pt`, which holds EMA weights and
    # no optimizer state): the weights are loaded before training, the optimizer and EMA start fresh
    # and the lr warms up again. `init_checkpoint_steps` is the optimizer step that checkpoint had
    # reached; it offsets the logged step so the W&B curve continues the original run's x-axis.
    init_checkpoint: Optional[str] = None
    init_checkpoint_steps: int = 0
    # Log into the W&B run the checkpoint came from instead of a new one (wandb.init(resume="must")),
    # so the continued training shows as one curve; needs `init_checkpoint_steps` so steps carry on.
    wandb_resume_id: Optional[str] = None

    # Loss weight on the FINAL readout of a deeply supervised model; the remaining `1 - w` is spread
    # evenly over the earlier readouts. None (default) weights every readout equally, i.e. w = 1/R.
    # w = 1.0 supervises only the final readout, which is what `rt@RecurrentTransformer` does.
    final_readout_weight: Optional[float] = None

# [Utils]
# Top-level directories never worth shipping to wandb as source code: the virtualenvs (the big
# one -- `.venv` and `.venv-vllm` hold ~18 GB of site-packages *.py between them), the datasets,
# and anything this repo writes itself.
_CODE_ARTIFACT_SKIP = (".venv", "venv", ".git", "wandb", "logs", "outputs", "checkpoints",
                       "downloaded-datasets", "node_modules", "__pycache__")

def _exclude_from_code_artifact(path: str, root: str = ".") -> bool:
    """True if `path` should be kept out of the wandb code artifact."""
    rel = os.path.relpath(path, root)
    return any(part.startswith(_CODE_ARTIFACT_SKIP) for part in rel.split(os.sep))

def load_module(identifier: str):
    module_path, class_name = identifier.split('@')
    # Import the module
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# [Training and Inference Step]
def model_input(x: Tensor, y: Tensor, is_autoregressive: bool) -> Tensor:
    """Decoder-only models are teacher-forced on `[question ++ answer]`; encoders only see the question."""
    return torch.cat([x, y], dim=-1) if is_autoregressive else x

def readouts(y_hat: Tensor) -> Tensor:
    """Normalise model outputs to `[batch, num_readouts, seq_len, vocab]`.

    Deeply supervised models return one readout per supervision point; every other model returns a
    single `[batch, seq_len, vocab]` set of logits. The last readout is always the final prediction.
    """
    return y_hat if y_hat.ndim == 4 else y_hat.unsqueeze(1)

def readout_loss(logits: Tensor, y: Tensor) -> Tensor:
    """Mean cross-entropy (in f32) of `[batch, num_readouts, seq_len, vocab]` against `y`."""
    targets = y.unsqueeze(1).expand(-1, logits.shape[1], -1)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).to(torch.float32), targets.reshape(-1).long(), reduction="mean")

def train_step(model: nn.Module, carry: Carry, opt: torch.optim.Optimizer, x: Tensor, y: Tensor, is_autoregressive: bool = False, final_readout_weight: Optional[float] = None):
    carry, y_hat = model(carry, model_input(x, y, is_autoregressive))
    # loss (f32 for CrossEntropy), averaged over the supervision points of deeply supervised models
    logits = readouts(y_hat)
    if final_readout_weight is None or logits.shape[1] == 1:
        loss = readout_loss(logits, y)
    else:
        # Put `w` on the final readout and spread `1 - w` over the earlier ones. w = 1/R reproduces
        # the equal-weight average above; w = 1 supervises only the final readout, as RT does.
        w = final_readout_weight
        loss = w * readout_loss(logits[:, -1:], y) + (1.0 - w) * readout_loss(logits[:, :-1], y)
    loss.backward()
    opt.step()
    opt.zero_grad()

    # metrics (of the final readout, i.e. what inference uses)
    with torch.no_grad():
        preds = torch.argmax(logits[:, -1], dim=-1)
        metrics = {
            "loss": loss.detach(),
            "per_position_accuracy": torch.mean(preds == y, dtype=torch.float32),
            "exact_match": torch.mean(torch.all(preds == y, dim=-1), dtype=torch.float32)
        }

    return carry, metrics

@torch.inference_mode()
def run_inference(model: nn.Module, carry: Carry, x: Tensor):
    carry, y_hat = model(carry, x)
    return carry, torch.argmax(readouts(y_hat)[:, -1], dim=-1)

@torch.inference_mode()
def generate(model: nn.Module, x: Tensor) -> Tensor:
    """Greedy decoding of the answer block for autoregressive models, one token at a time.

    The answer slots start as the BOS token (id 0) and are overwritten left to right; causal masking
    keeps the not-yet-decoded slots from leaking into earlier positions. Shapes stay static so the
    compiled graph is reused across all decoding steps.
    """
    block_len = x.shape[-1]
    seq = torch.cat([x, torch.zeros_like(x)], dim=-1)
    for pos in range(1, block_len):
        _carry, y_hat = model({}, seq)
        seq[:, block_len + pos] = torch.argmax(y_hat[:, pos], dim=-1)
    return seq[:, block_len:]

def update_lr(config: TrainConfig, optim: torch.optim.Optimizer, step: int, total_steps: int, anneal_start: Optional[int] = None) -> float:
    # Linear warmup cosine schedule
    if step < config.lr_warmup_steps:
        lr = config.lr * min(1.0, step / config.lr_warmup_steps)
    else:
        progress = (step - config.lr_warmup_steps) / (total_steps - config.lr_warmup_steps)
        lr = config.lr * (config.lr_min_ratio + max(0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))
    # Plateau-triggered anneal (train-to-convergence): cosine from the current lr down to
    # anneal_min_ratio x over anneal_steps, starting at the step the plateau was detected.
    if anneal_start is not None:
        progress = min(1.0, max(0.0, (step - anneal_start) / config.anneal_steps))
        lr *= config.anneal_min_ratio + (1 - config.anneal_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optim.param_groups:
        param_group["lr"] = torch.tensor(lr * param_group.get("lr_mult", 1.0),
                                         dtype=torch.get_default_dtype(), device="cpu")

    return lr


def train_single_seed(config: TrainConfig, seed: int, group_name: str, WORLD_SIZE: int, RANK: int):
    """Run a full training run for a single seed."""
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize Dataloader
    create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
    train_loader, train_metadata = create_dataloader("train", config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, **config.data.__pydantic_extra__)  # pyright: ignore[reportCallIssue]
    eval_data_kwargs = dict(config.data.__pydantic_extra__ or {}) | (
        {"dataset_name": config.eval_dataset_name} if config.eval_dataset_name else {})
    eval_loaders = {split_name: create_dataloader(split_name, config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, **eval_data_kwargs)[0] for split_name in config.eval_splits}  # pyright: ignore[reportCallIssue]

    total_steps = int(config.cycles_per_data * len(train_loader) * config.epochs)
    if config.max_steps is not None:
        if config.max_steps > total_steps:
            raise ValueError(f"max_steps={config.max_steps} exceeds the {total_steps} steps that epochs={config.epochs} provides")
        total_steps = config.max_steps

    # Initialize Model and Optimizer
    model_cls = load_module(f"arch.{config.arch.name}")
    is_autoregressive: bool = getattr(model_cls, "is_autoregressive", False)
    with torch.device("cuda"):
        model: nn.Module = model_cls(config.arch.__pydantic_extra__ | train_metadata)
        if config.init_checkpoint:
            model.load_state_dict(torch.load(config.init_checkpoint, map_location="cuda", weights_only=True))
            if RANK == 0:
                print(f"[init] loaded weights from {config.init_checkpoint} (trained for {config.init_checkpoint_steps} steps)", flush=True)
        model = torch.compile(model, dynamic=False, fullgraph=True)  # pyright: ignore[reportAssignmentType]

        # DDP Wrap
        model = DDP(model, static_graph=True)

    # Frozen arms (e.g. a new module on a frozen backbone) leave most parameters without a
    # gradient; keeping them out of the optimizer avoids allocating momentum/EMA buffers for them.
    # A model may also split its parameters into groups carrying an `lr_mult`, so that a newly
    # added component can train at a different rate from a pretrained backbone (see `update_lr`).
    if hasattr(model.module, "param_groups"):
        param_groups = model.module.param_groups()
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr_mult": 1.0}]
    optim = AdamATan2(
        param_groups,
        lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        ema=config.ema
    )

    # Initialize checkpointing (rank 0 only, all ranks hold identical weights)
    checkpoint_dir = os.path.join("checkpoints", group_name, f"seed_{seed}")
    if RANK == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

        with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
            yaml.dump(config.model_dump(), f)

    # -----Train & Eval loop
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=total_steps, desc=f"seed={seed}")

        # Same wandb run name across seeds so runs can be grouped by name
        if config.wandb_resume_id:
            # Continue the original run: same id, config updated with the continuation's settings.
            wandb.init(project=config.data.name, id=config.wandb_resume_id, resume="must",
                       config=config.model_dump() | {"seed": seed}, allow_val_change=True,
                       settings=wandb.Settings(x_disable_stats=True))
        else:
            wandb.init(project=config.data.name,
                       name=group_name,
                       group=group_name,
                       config=config.model_dump() | {"seed": seed},
                       settings=wandb.Settings(x_disable_stats=True))
        if wandb.run is not None:
            # Bare log_code() walks every *.py under the repo root, which sweeps `.venv/` and
            # `.venv-vllm/` (~18 GB of site-packages) into wandb's artifact staging area on every
            # seed of every run and fills the disk mid-training. Keep the repo's own sources and
            # skip the environments, data and output dirs. `exclude_fn` is called as (path, root).
            wandb.run.log_code(exclude_fn=_exclude_from_code_artifact)

    step = 0
    best_metric = -1.0

    def evaluate_and_checkpoint(step: int):
        """Eval every split, log it, and write last.pt / best.pt. Collective: every rank must call
        this at the same step, which holds because `step` advances identically on all ranks."""
        nonlocal best_metric
        model.eval()
        optim.swap_ema()

        eval_metrics = {}
        for eval_name, eval_loader in eval_loaders.items():
            num_total_correct = torch.zeros(2, dtype=torch.long, device="cuda")
            for x, y in eval_loader:
                # Run inference. Autoregressive models must decode the answer instead of reading it.
                if is_autoregressive:
                    y_hat = generate(model, x.cuda())
                else:
                    carry: Carry = model.module.initial_carry
                    y_hat = None
                    for _ in range(config.cycles_per_data):
                        carry, y_hat = run_inference(model, carry, x.cuda())
                    del carry

                num_total_correct[0] += torch.all(y_hat == y.cuda(), dim=-1).sum()
                num_total_correct[1] += y.shape[0]

                del y_hat

            # Reduce and log
            dist.reduce(num_total_correct, dst=0)
            num_total_correct = num_total_correct.cpu().tolist()
            if RANK == 0:
                exact_match = num_total_correct[0] / num_total_correct[1]
                eval_metrics[eval_name] = exact_match
                wandb.log({f"eval/{eval_name}_exact_match": exact_match}, step=step + config.init_checkpoint_steps)
                print(f"[eval] step {step + config.init_checkpoint_steps} {eval_name} exact_match {exact_match:.5f} "
                      f"({num_total_correct[0]}/{num_total_correct[1]})", flush=True)

        improved = False
        # Save model (rank 0 only, all ranks hold identical weights).
        # Only 'last' and 'best' are kept; both are saved with EMA weights swapped in,
        # i.e. exactly the weights that produced the eval numbers above.
        # Clean '_orig_mod.' prefix added by torch.compile for easier downstream loading
        if RANK == 0:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in model.module.state_dict().items()}
            torch.save(state_dict, os.path.join(checkpoint_dir, "last.pt"))

            score = eval_metrics.get(config.best_metric_split)
            if score is not None and score > best_metric:
                best_metric = score
                improved = True
                torch.save(state_dict, os.path.join(checkpoint_dir, "best.pt"))

            del state_dict

        optim.swap_ema()  # Swap EMA back

        # Also keep the raw (non-EMA) weights at the same points, so both can be evaluated without
        # retraining; `best_raw.pt` follows the EMA-selected best step.
        if RANK == 0 and config.ema is not None:
            raw_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model.module.state_dict().items()}
            torch.save(raw_state_dict, os.path.join(checkpoint_dir, "last_raw.pt"))
            if improved:
                torch.save(raw_state_dict, os.path.join(checkpoint_dir, "best_raw.pt"))
            del raw_state_dict
        model.train()

    last_eval_step = -1
    next_eval_step = config.eval_interval if config.eval_interval is not None else None

    # Train-to-convergence controller (see TrainConfig.plateau_patience_steps). Rank 0 tracks the best
    # score and decides; `anneal_start` and the stop flag are broadcast so every rank keeps the same
    # lr schedule and leaves the loop at the same step.
    plateau_best, plateau_best_step = -1.0, 0
    anneal_start: Optional[int] = None
    stop_training = False

    def plateau_decision(step: int):
        nonlocal plateau_best, plateau_best_step, anneal_start, stop_training
        if config.plateau_patience_steps is None and config.early_stop_patience_steps is None:
            return
        state = torch.tensor([-1, 0], dtype=torch.long, device="cuda")
        if RANK == 0:
            if best_metric > plateau_best + config.plateau_min_delta:
                plateau_best, plateau_best_step = best_metric, step
            if config.early_stop_patience_steps is not None and step - plateau_best_step >= config.early_stop_patience_steps:
                stop_training = True
                print(f"[early-stop] best {plateau_best:.4f} at step {plateau_best_step}; no gain >= {config.plateau_min_delta} for "
                      f"{step - plateau_best_step} steps -> stopping at step {step}", flush=True)
            if config.plateau_patience_steps is not None and anneal_start is None and step - plateau_best_step >= config.plateau_patience_steps:
                anneal_start = step
                print(f"[plateau] best {plateau_best:.4f} at step {plateau_best_step}; no gain >= {config.plateau_min_delta} for "
                      f"{step - plateau_best_step} steps -> annealing lr to {config.anneal_min_ratio}x over {config.anneal_steps} steps", flush=True)
                wandb.log({"train/anneal_start": anneal_start + config.init_checkpoint_steps}, step=step + config.init_checkpoint_steps)
            if anneal_start is not None and step >= anneal_start + config.anneal_steps:
                stop_training = True
                print(f"[plateau] anneal finished at step {step}; stopping (best {best_metric:.4f})", flush=True)
            state[0] = -1 if anneal_start is None else anneal_start
            state[1] = int(stop_training)
        if dist.is_initialized():
            dist.broadcast(state, src=0)
        anneal_start = None if state[0].item() < 0 else int(state[0].item())
        stop_training = bool(state[1].item())

    for epoch in range(config.epochs):
        model.train()
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()

            metrics = {}
            lr = None
            carry: Carry = model.module.initial_carry
            for _ in range(config.cycles_per_data):
                step += 1
                lr = update_lr(config, optim, step, total_steps, anneal_start)

                carry, metrics = train_step(model, carry, optim, x, y, is_autoregressive, config.final_readout_weight)

            if RANK == 0 and progress_bar is not None and step - progress_bar.n >= config.log_interval:
                progress_bar.update(step - progress_bar.n)
                wandb.log({f"train/{k}": v.item() for k, v in metrics.items()} | {"train/lr": lr}, step=step + config.init_checkpoint_steps)

            del x, y, carry, metrics

            # Step-based eval, when configured (see `eval_interval`)
            if next_eval_step is not None and step >= next_eval_step:
                evaluate_and_checkpoint(step)
                last_eval_step = step
                while next_eval_step <= step:
                    next_eval_step += config.eval_interval
                plateau_decision(step)
                if stop_training:
                    break
            if config.max_steps is not None and step >= total_steps:
                stop_training = True
                break

        if stop_training:
            break

        # Per-epoch eval, the default when `eval_interval` is unset
        if config.eval_interval is None:
            evaluate_and_checkpoint(step)
            last_eval_step = step
            plateau_decision(step)
            if stop_training:
                break

    # Always finish on an eval of the final weights
    if last_eval_step != step:
        evaluate_and_checkpoint(step)

    # Close progress bar and wandb run for this seed
    if progress_bar is not None:
        progress_bar.close()
    if RANK == 0:
        wandb.finish()


# [Training Loop]
@hydra.main(config_path="config", version_base=None)
def train(config_dict: dict[str, Any]):
    WORLD_SIZE = 1
    RANK = 0
    DEVICE_ID = 0

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
        DEVICE_ID = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(DEVICE_ID)

    # Load config
    config = TrainConfig(**config_dict)

    # Generate a shared group name for all seeds in this run
    group_name = os.environ.get("MLP_TASK_NAME", f"{config.run_name or HydraConfig.get().job.config_name} {coolname.generate_slug(2)}")

    for seed in config.seeds:
        train_single_seed(config, seed, group_name, WORLD_SIZE, RANK)

if __name__ == "__main__":
    train()
