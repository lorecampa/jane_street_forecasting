class ConfigOAMP:
    def __init__(
        self,
        args: dict,
    ):
        self.agents_weights_upd_freq = args.get("agents_weights_upd_freq", 10)
        self.loss_fn_window = args.get("loss_fn_window", 10)
        self.agg_type = args.get("agg_type", "mean")
