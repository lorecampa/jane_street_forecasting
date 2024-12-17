import lightning.pytorch.callbacks as C

class EpochStatsCallback(C.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics  # Get all logged metrics
        train_metrics = {k: v.item() for k, v in metrics.items() if k.startswith("train_")}
        epoch = trainer.current_epoch
        print(f"\n[Epoch {epoch} - Training]")
        for key, value in train_metrics.items():
            print(f"{key}: {value:.4f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics  # Get all logged metrics
        val_metrics = {k: v.item() for k, v in metrics.items() if k.startswith("val_")}
        epoch = trainer.current_epoch
        print(f"\n[Epoch {epoch} - Validation]")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

