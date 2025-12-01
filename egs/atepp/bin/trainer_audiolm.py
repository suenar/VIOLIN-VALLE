#!/usr/bin/env python3
# Copyright    2024                             (authors: Your Name)
#
# Training script for AudioLM model
# This script shows the key modifications needed to train the audio-only model

"""
Usage example:

python3 bin/trainer_audiolm.py \
    --model-name VALLE-audio --num-buckets 12 --save-every-n 20000\
    --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
    --base-lr 0.05 --warmup-steps 200 \
    --num-epochs 20 --start-epoch 1 \
    --exp-dir exp/VALLE-audio \
    --dataset atepp --max-duration 150 \
    --num-quantizers 4 \
    --train-stage 0 \
    --manifest-dir "data/tokenized"\
    --filter-min-duration 8 --inf-check True\
    --filter-max-duration 20 --world-size 1
"""

import sys
sys.path.insert(0, '/workspace')

# Import the standard trainer
from egs.atepp.bin.trainer import *


# Override the compute_loss function for AudioLM
def compute_loss_audiolm(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss for AudioLM model.
    
    Key differences from standard VALLE:
    - Input x: prompt audio tokens (conditioning, e.g., 3-5 seconds)
    - Input y: target audio tokens (what to predict, e.g., next 5-10 seconds)
    - Still uses AR (1st quantizer) + NAR (remaining quantizers) structure
    
    To prepare data:
    - Take a long audio sequence (e.g., 10-15 seconds)
    - Split into prompt (first 3-5 seconds) and target (remaining)
    - x = prompt audio tokens
    - y = target audio tokens
    """
    device = (
        model.device
        if isinstance(model, DDP)
        else next(model.parameters()).device
    )
    
    # Get audio features
    audio_features = batch["audio_features"].to(device)
    audio_features_lens = batch["audio_features_lens"].to(device)
    assert audio_features.ndim == 3  # (B, T, Q)
    
    # Split audio into prompt and target
    # For training, we can use a sliding window or fixed split
    # Here we use first 40% as prompt, rest as target
    batch_size, total_len, num_q = audio_features.shape
    
    # Calculate split point for each sample in batch
    split_points = (audio_features_lens * 0.4).long()
    split_points = torch.clamp(split_points, min=50, max=total_len // 2)
    
    # For simplicity, use the minimum split point for the whole batch
    # In practice, you might want to handle variable lengths more carefully
    split_point = split_points.min().item()
    
    # Split into prompt (x) and target (y)
    x = audio_features[:, :split_point, :]
    y = audio_features[:, split_point:, :]
    
    # Adjust lengths
    x_lens = torch.full_like(audio_features_lens, split_point)
    x_lens = torch.clamp(x_lens, max=split_point)
    
    y_lens = audio_features_lens - split_point
    y_lens = torch.clamp(y_lens, min=1)
    
    with torch.set_grad_enabled(is_training):
        # Forward pass - model expects:
        # - x: prompt audio tokens (B, S, Q)
        # - x_lens: prompt lengths (B,)
        # - y: target audio tokens (B, T, Q)
        # - y_lens: target lengths (B,)
        predicts, loss, metrics = model(
            x=x,
            x_lens=x_lens,
            y=y,
            y_lens=y_lens,
            reduction="sum",
            train_stage=params.train_stage,
        )
    
    assert loss.requires_grad == is_training
    
    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = y_lens.sum().item()
        info["utterances"] = audio_features.size(0)
    
    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    for metric in metrics:
        info[metric] = metrics[metric].detach().cpu().item()
    del metrics
    
    return predicts, loss, info


def compute_validation_loss_audiolm(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process for AudioLM."""
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        predicts, loss, loss_info = compute_loss_audiolm(
            params=params,
            model=model,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info
    
    if world_size > 1:
        tot_loss.reduce(loss.device)
    
    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


# Override train_one_epoch to use the new compute_loss
def train_one_epoch_audiolm(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    rng: random.Random,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Modified train_one_epoch that uses compute_loss_audiolm."""
    model.train()
    tot_loss = MetricsTracker()
    iter_dl = iter(train_dl)

    dtype, enabled = torch.float32, False
    if params.dtype in ["bfloat16", "bf16"]:
        dtype, enabled = torch.bfloat16, True
    elif params.dtype in ["float16", "fp16"]:
        dtype, enabled = torch.float16, True

    batch_idx = 0
    while True:
        try:
            batch = next(iter_dl)
        except StopIteration:
            logging.info("Reaches end of dataloader.")
            break

        batch_idx += 1
        params.batch_idx_train += 1
        batch_size = len(batch["audio_features"])

        try:
            with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                _, loss, loss_info = compute_loss_audiolm(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            
            # summary stats
            tot_loss = (
                tot_loss * (1 - 1 / params.reset_interval)
            ) + loss_info * (1 / params.reset_interval)

            scaler.scale(loss).backward()
            if params.batch_idx_train >= params.accumulate_grad_steps:
                if (
                    params.batch_idx_train % params.accumulate_grad_steps
                    == 0
                ):
                    if params.optimizer_name not in ["ScaledAdam", "Eve"]:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    for k in range(params.accumulate_grad_steps):
                        if isinstance(scheduler, Eden):
                            scheduler.step_batch(params.batch_idx_train)
                        else:
                            scheduler.step()

            set_batch_count(model, params.batch_idx_train)
        except:  # noqa
            display_and_save_batch(batch, params=params)
            raise

        if params.average_period > 0:
            if (
                params.batch_idx_train > 0
                and params.batch_idx_train % params.average_period == 0
            ):
                if rank == 0:
                    update_averaged_model(
                        params=params,
                        model_cur=model,
                        model_avg=model_avg,
                    )
             
        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            if rank == 0:
                save_checkpoint_with_global_batch_idx(
                    out_dir=params.exp_dir,
                    global_batch_idx=params.batch_idx_train,
                    model=model,
                    model_avg=model_avg,
                    params=params,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=train_dl.sampler,
                    scaler=scaler,
                    rank=rank,
                )
                remove_checkpoints(
                    out_dir=params.exp_dir,
                    topk=params.keep_last_k,
                    rank=rank,
                )

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            cur_grad_scale = (
                scaler._scale.item()
                if params.dtype in ["float16", "fp16"]
                else 1.0
            )

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, train_loss[{loss_info}], "
                f"tot_loss[{tot_loss}], "
                f"batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer,
                    "train/current_",
                    params.batch_idx_train,
                )
                tot_loss.write_summary(
                    tb_writer, "train/tot_", params.batch_idx_train
                )

        if params.batch_idx_train % params.valid_interval == 0:
            model.eval()
            logging.info("Computing validation loss")
            with torch.cuda.amp.autocast(dtype=dtype):
                valid_info = compute_validation_loss_audiolm(
                    params=params,
                    model=model,
                    valid_dl=valid_dl,
                    world_size=world_size,
                )
            logging.info(
                f"Epoch {params.cur_epoch}, validation: {valid_info}"
            )

            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

            model.train()

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


# Override run function to use modified training
def run_audiolm(rank, world_size, args):
    """Modified run function for AudioLM training."""
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    rng = random.Random(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training AudioLM started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create AudioLM model")
    model = get_model(params)
    
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    model_avg: Optional[nn.Module] = None
    if rank == 0 and params.average_period > 0:
        model_avg = copy.deepcopy(model).to(torch.float64)

    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Setup optimizer
    # if params.model_name.lower() in ["valle-audio", "valle_audio"]:
    #     if params.train_stage:
    #         _model = model.valle-audio.module if isinstance(model.valle-audio, DDP) else model.valle-audio
    #         model_parameters = _model.stage_parameters(params.train_stage)
    #     else:
    #         model_parameters = model.valle-audio.parameters()
    # else:     
    #     if params.train_stage:
    #         _model = model.module if isinstance(model, DDP) else model
    #         model_parameters = _model.stage_parameters(params.train_stage)
    #     else:
    #         model_parameters = model.parameters()

    if params.optimizer_name == "ScaledAdam":
        parameters_names = []
        if params.model_name.lower() in ["valle-audio", "valle_audio"]:
            if params.train_stage:  # != 0
                _model = model.valle.module if isinstance(model.valle, DDP) else model.valle
                parameters_names.append(
                    [
                        name_param_pair[0]
                        for name_param_pair in _model.stage_named_parameters(
                            params.train_stage
                        )
                    ]
                )
            else:
                parameters_names.append(
                    [
                        name_param_pair[0]
                        for name_param_pair in model.named_parameters()
                    ]
                )
        else:
            if params.train_stage:  # != 0
                _model = model.module if isinstance(model, DDP) else model
                parameters_names.append(
                    [
                        name_param_pair[0]
                        for name_param_pair in _model.stage_named_parameters(
                            params.train_stage
                        )
                    ]
                )
            else:
                parameters_names.append(
                    [
                        name_param_pair[0]
                        for name_param_pair in model.named_parameters()
                    ]
                )
        optimizer = ScaledAdam(
            model.parameters(),
            lr=params.base_lr,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )
    elif params.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.base_lr,
            betas=(0.9, 0.95),
            weight_decay=1e-2,
        )
    else:
        raise NotImplementedError()

    scheduler = get_scheduler(params, optimizer)
    optimizer.zero_grad()

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if checkpoints and "scheduler" in checkpoints:
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    # Load data
    dataset = TtsDataModule(args)
    train_cuts = dataset.train_cuts()
    valid_cuts = dataset.dev_cuts()

    train_cuts = filter_short_and_long_utterances(
        train_cuts, params.filter_min_duration, params.filter_max_duration
    )
    valid_cuts = filter_short_and_long_utterances(
        valid_cuts, params.filter_min_duration, params.filter_max_duration
    )

    train_dl = dataset.train_dataloaders(train_cuts)
    valid_dl = dataset.valid_dataloaders(valid_cuts)

    scaler = GradScaler(enabled=(params.dtype in ["fp16", "float16"]))
    
    if checkpoints and "grad_scaler" in checkpoints:
        scaler.load_state_dict(checkpoints["grad_scaler"])

    # Training loop
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        if isinstance(scheduler, Eden):
            scheduler.step_epoch(epoch - 1)

        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        train_one_epoch_audiolm(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            rng=rng,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        cleanup_dist()


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    
    if world_size > 1:
        mp.spawn(run_audiolm, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_audiolm(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
