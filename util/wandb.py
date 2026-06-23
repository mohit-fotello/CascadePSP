import numpy as np

from util.image_saver import inv_im_trans, inv_seg_trans


class WandbLogger:
    def __init__(self, args, run_name, config):
        self.run = None
        self.wandb = None

        if not args.wandb:
            return

        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Install wandb or run with --no_wandb.") from exc

        self.wandb = wandb
        self.run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config=config,
        )

    @property
    def enabled(self):
        return self.run is not None

    def log_integrator(self, integrator, prefix, step, lr=None):
        if not self.enabled:
            return

        metrics = {}
        if lr is not None:
            metrics["train/lr"] = lr
        for key, value in integrator.values.items():
            metrics["%s/%s" % (prefix, key)] = value / integrator.counts[key]
        self.run.log(metrics, step=step)

    def log_image(self, key, image, step):
        if not self.enabled:
            return

        self.run.log({key: self.wandb.Image(image)}, step=step)

    def log_validation_table(self, rows, step):
        if not self.enabled:
            return

        table = self.wandb.Table(columns=["input", "prediction", "ground_truth"])
        for row in rows:
            table.add_data(
                self.wandb.Image(row["image"]),
                self.wandb.Image(row["prediction"]),
                self.wandb.Image(row["gt"]),
            )

        self.run.log({"val_samples_iter_%s" % step: table}, step=step)

    def finish(self):
        if self.enabled:
            self.run.finish()


def tensor_to_image(tensor):
    tensor = inv_im_trans(tensor.detach().cpu()).clamp(0, 1)
    array = tensor.numpy().transpose((1, 2, 0))
    return (array * 255).astype(np.uint8)


def tensor_to_mask(tensor, invert_normalization=False):
    tensor = tensor.detach().cpu()
    if invert_normalization:
        tensor = inv_seg_trans(tensor)
    tensor = tensor.clamp(0, 1)
    array = tensor.numpy()
    if array.ndim == 3:
        array = array[0]
    return (array * 255).astype(np.uint8)
