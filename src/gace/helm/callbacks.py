from lightning.pytorch.callbacks import Callback
from torch_ema import ExponentialMovingAverage


class EMACallback(Callback):
    """
    Maintains an exponential moving average of model weights during training.

    EMA weights are swapped in during validation so val/loss reflects EMA
    quality, then restored so training continues from the live weights.
    The EMA state is persisted in checkpoints for seamless resumption.

    Args:
        decay: EMA decay factor. Higher values track changes more slowly.
               Typical range: 0.999–0.9999.
    """

    def __init__(self, decay: float = 0.9999):
        self.decay = decay
        self.ema: ExponentialMovingAverage | None = None

    @staticmethod
    def _ema_target(pl_module):
        """The parameters EMA should track: the oracle_model when training
        it standalone (oracle_mode == 'train'), otherwise the flow model."""
        if getattr(pl_module, "oracle_mode", "off") == "train":
            return pl_module.oracle_model.parameters()
        return pl_module.model.parameters()

    def on_fit_start(self, trainer, pl_module):
        self.ema = ExponentialMovingAverage(
            self._ema_target(pl_module), decay=self.decay
        )

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        # Called after each optimizer step — update shadow weights.
        self.ema.update()

    def on_validation_start(self, trainer, pl_module):
        # Stash live weights; copy EMA weights in for validation.
        self.ema.store(self._ema_target(pl_module))
        self.ema.copy_to(self._ema_target(pl_module))

    def on_validation_end(self, trainer, pl_module):
        # Restore live weights so training continues correctly.
        self.ema.restore(self._ema_target(pl_module))

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state_dict" in checkpoint:
            if self.ema is None:
                self.ema = ExponentialMovingAverage(
                    self._ema_target(pl_module), decay=self.decay
                )
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
