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
        val_metrics['val_wmse'] = pl_module.acc_metrics['ss_res'] / (pl_module.acc_metrics['weights_sum'] + 1e-10)
        val_metrics['val_wmae'] = pl_module.acc_metrics['abs_err_sum'] / (pl_module.acc_metrics['weights_sum'] + 1e-10)
        val_metrics['val_wr2'] = 1 - pl_module.acc_metrics['ss_res'] / (pl_module.acc_metrics['ss_tot'] + 1e-10)
        # pl_module.acc_metrics = dict(ss_res=0.0, ss_tot=0.0, abs_err_sum=0.0, weights_sum=0.0)
        epoch = trainer.current_epoch
        print(f"\n[Epoch {epoch} - Validation]")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

