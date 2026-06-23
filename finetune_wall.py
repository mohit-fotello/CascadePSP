import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

from dataset import WallRefinementDataset
from models.psp.pspnet import PSPNet
from models.sobel_op import SobelComputer
from util.hyper_para import HyperParameters
from util.image_saver import vis_prediction
from util.log_integrator import Integrator
from util.logger import BoardLogger
from util.metrics_compute import compute_loss_and_metrics, iou_hooks_to_be_used
from util.model_saver import ModelSaver
from util.wandb import WandbLogger, tensor_to_image, tensor_to_mask


DEFAULT_PRETRAINED = os.path.expanduser("~/.segmentation-refinement/model")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CascadePSP on one image/mask dataset.")
    parser.add_argument("id", help="Experiment id. Use NULL to disable tensorboard logging/saving.")
    parser.add_argument("--data", default=os.path.join("data", "walls"), help="Dataset root folder.")
    parser.add_argument("--image_dir", default="images", help="Image folder, relative to --data unless absolute.")
    parser.add_argument("--raw_mask_dir", default="raw_masks", help="BiRefNet raw-mask folder, relative to --data unless absolute.")
    parser.add_argument("--gt_mask_dir", default="gt_masks", help="Ground-truth mask folder, relative to --data unless absolute.")
    parser.add_argument("--load", default=DEFAULT_PRETRAINED, help="Pretrained CascadePSP checkpoint path.")
    parser.add_argument("-i", "--iterations", default=50000, type=int, help="Number of fine-tuning iterations.")
    parser.add_argument("-b", "--batch_size", default=2, type=int, help="Batch size.")
    parser.add_argument("--lr", default=2e-5, type=float, help="Fine-tuning learning rate.")
    parser.add_argument("--steps", default=None, type=int, nargs="*", help="LR decay steps. Defaults to 50%% and 80%% of iterations.")
    parser.add_argument("--gamma", default=0.1, type=float, help="LR decay multiplier.")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Adam weight decay.")
    parser.add_argument("--num_workers", default=8, type=int, help="DataLoader worker count.")
    parser.add_argument("--report_interval", default=50, type=int, help="Metric logging interval.")
    parser.add_argument("--save_im_interval", default=500, type=int, help="Tensorboard prediction image interval.")
    parser.add_argument("--val_interval", default=200, type=int, help="Validation and W&B table logging interval.")
    parser.add_argument("--val_split", default=0.05, type=float, help="Validation fraction.")
    parser.add_argument("--seed", default=42, type=int, help="Train/val split seed.")
    parser.add_argument("--gpus", default="0", help="Comma-separated CUDA ids. Example: 0 or 0,1.")
    parser.add_argument("--no_compile", action="store_false", dest="compile", default=True, help="Disable torch.compile.")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb", default=True, help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="cascadepsp-wall-finetune", help="W&B project name.")
    parser.add_argument("--wandb_entity", default=None, help="W&B entity/team.")
    parser.add_argument("--wandb_mode", default=None, choices=["online", "offline", "disabled"], help="W&B run mode.")
    return parser.parse_args()


def make_loss_params(args):
    params = HyperParameters()
    params.args = {
        "ce_weight": [0.0, 1.0, 0.5, 1.0, 1.0, 0.5],
        "l1_weight": [1.0, 0.0, 0.25, 0.0, 0.0, 0.25],
        "l2_weight": [1.0, 0.0, 0.25, 0.0, 0.0, 0.25],
        "grad_weight": 5,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
        "steps": args.steps,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "load": args.load,
        "id": args.id,
        "compile": args.compile,
        "val_interval": args.val_interval,
        "val_split": args.val_split,
        "seed": args.seed,
        "data": args.data,
        "image_dir": args.image_dir,
        "raw_mask_dir": args.raw_mask_dir,
        "gt_mask_dir": args.gt_mask_dir,
    }
    return params


def load_state_dict(model, checkpoint_path):
    if not checkpoint_path:
        return

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found: %s" % checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_keys = next(iter(model.state_dict().keys()))
    ckpt_keys = next(iter(state_dict.keys()))

    if model_keys.startswith("module.") and not ckpt_keys.startswith("module."):
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif not model_keys.startswith("module.") and ckpt_keys.startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print("Loaded checkpoint from %s." % checkpoint_path)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def make_splits(args):
    train_full = WallRefinementDataset(
        args.data,
        image_dir=args.image_dir,
        raw_mask_dir=args.raw_mask_dir,
        gt_mask_dir=args.gt_mask_dir,
        perturb=True,
    )
    val_full = WallRefinementDataset(
        args.data,
        image_dir=args.image_dir,
        raw_mask_dir=args.raw_mask_dir,
        gt_mask_dir=args.gt_mask_dir,
        perturb=False,
    )

    dataset_size = len(train_full)
    if dataset_size < 2:
        raise RuntimeError("Need at least 2 image/mask pairs for a train/val split.")

    val_count = max(1, int(round(dataset_size * args.val_split)))
    val_count = min(val_count, dataset_size - 1)
    indices = np.random.default_rng(args.seed).permutation(dataset_size).tolist()

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return Subset(train_full, train_indices), Subset(val_full, val_indices)


def get_integrator_average(integrator, key):
    if key not in integrator.values:
        return None
    return integrator.values[key] / integrator.counts[key]


def run_validation(model, val_loader, sobel_compute, params, logger, wandb_logger, step, device_id):
    model.eval()
    val_integrator = Integrator(logger)
    val_integrator.add_hook(iou_hooks_to_be_used)
    first_images = None
    table_rows = []

    with torch.no_grad():
        for im, seg, gt in val_loader:
            im = im.cuda(device_id, non_blocking=True)
            seg = seg.cuda(device_id, non_blocking=True)
            gt = gt.cuda(device_id, non_blocking=True)

            images = model(im, seg)
            images["im"] = im
            images["seg"] = seg
            images["gt"] = gt

            sobel_compute.compute_edges(images)
            loss_and_metrics = compute_loss_and_metrics(images, params)
            val_integrator.add_dict(loss_and_metrics)

            if first_images is None:
                first_images = images

            batch_size = im.shape[0]
            for sample_idx in range(batch_size):
                table_rows.append({
                    "image": tensor_to_image(im[sample_idx]),
                    "prediction": tensor_to_mask(images["pred_224"][sample_idx]),
                    "gt": tensor_to_mask(gt[sample_idx]),
                    "seg": tensor_to_mask(seg[sample_idx], invert_normalization=True),
                })

    val_integrator.finalize("val", step)
    wandb_logger.log_integrator(val_integrator, "val", step)
    wandb_logger.log_validation_table(table_rows, step)

    if first_images is not None:
        predict_vis = vis_prediction(first_images)
        logger.log_cv2("val/predict", predict_vis, step)
        wandb_logger.log_image("val/predict", predict_vis, step)

    return get_integrator_average(val_integrator, "total_loss")


def main():
    args = parse_args()
    if args.steps is None:
        args.steps = [args.iterations // 2, int(args.iterations * 0.8)]

    if not torch.cuda.is_available():
        raise RuntimeError("This fine-tuning script requires CUDA because the existing Sobel loss uses .cuda().")

    torch.backends.cudnn.benchmark = True
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",") if gpu_id.strip()]
    torch.cuda.set_device(gpu_ids[0])

    if args.id.lower() != "null":
        long_id = "%s_%s" % (args.id, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    else:
        long_id = None

    params = make_loss_params(args)
    logger = BoardLogger(long_id)
    logger.log_string("hyperpara", str(params))
    wandb_logger = WandbLogger(args, long_id or args.id, params.args)

    train_dataset, val_dataset = make_splits(args)
    print("Train dataset size: ", len(train_dataset))
    print("Val dataset size: ", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Dataset is too small for batch size %d." % args.batch_size)

    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend="resnet50")
    model = model.cuda(gpu_ids[0])
    load_state_dict(model, args.load)

    train_model = model
    if args.compile:
        train_model = torch.compile(train_model)
        print("Using torch.compile.")

    if len(gpu_ids) > 1:
        train_model = nn.DataParallel(train_model, device_ids=gpu_ids)

    optimizer = optim.Adam(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.steps, args.gamma)
    print("Using MultiStepLR with steps=%s gamma=%s." % (args.steps, args.gamma))
    sobel_compute = SobelComputer()
    saver = ModelSaver(long_id)

    train_integrator = Integrator(logger)
    train_integrator.add_hook(iou_hooks_to_be_used)

    total_iter = 0
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    while total_iter < args.iterations:
        np.random.seed()
        train_model.train()

        for im, seg, gt in train_loader:
            total_iter += 1
            im = im.cuda(gpu_ids[0], non_blocking=True)
            seg = seg.cuda(gpu_ids[0], non_blocking=True)
            gt = gt.cuda(gpu_ids[0], non_blocking=True)

            images = train_model(im, seg)
            images["im"] = im
            images["seg"] = seg
            images["gt"] = gt

            sobel_compute.compute_edges(images)
            loss_and_metrics = compute_loss_and_metrics(images, params)
            train_integrator.add_dict(loss_and_metrics)

            optimizer.zero_grad()
            loss_and_metrics["total_loss"].backward()
            optimizer.step()

            if total_iter % args.report_interval == 0:
                lr = scheduler.get_last_lr()[0]
                train_loss = get_integrator_average(train_integrator, "total_loss")
                logger.log_scalar("train/lr", lr, total_iter)
                train_integrator.finalize("train", total_iter)
                wandb_logger.log_integrator(train_integrator, "train", total_iter, lr)
                if train_loss is not None and train_loss < best_train_loss:
                    best_train_loss = train_loss
                    saver.save_named_model(model, "best_train_loss")
                train_integrator.reset_except_hooks()

            scheduler.step()

            if total_iter % args.save_im_interval == 0:
                predict_vis = vis_prediction(images)
                logger.log_cv2("train/predict", predict_vis, total_iter)
                wandb_logger.log_image("train/predict", predict_vis, total_iter)

            if total_iter % args.val_interval == 0:
                val_loss = run_validation(train_model, val_loader, sobel_compute, params, logger, wandb_logger, total_iter, gpu_ids[0])
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saver.save_named_model(model, "best_val_loss")
                train_model.train()

            if total_iter >= args.iterations:
                break

        saver.save_named_model(model, "last")

    if total_iter % args.val_interval != 0:
        val_loss = run_validation(train_model, val_loader, sobel_compute, params, logger, wandb_logger, total_iter, gpu_ids[0])
        if val_loss is not None and val_loss < best_val_loss:
            saver.save_named_model(model, "best_val_loss")
    saver.save_named_model(model, "last")
    wandb_logger.finish()


if __name__ == "__main__":
    main()
