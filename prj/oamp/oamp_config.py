class ConfigOAMP:
    def __init__(
        self,
        args: dict,
    ):
        self.agents_weights_upd_freq = args.get("agents_weights_upd_freq", 1)
        self.loss_fn_window = args.get("loss_fn_window", 1)
        self.action_thresh = args.get("action_thresh", 0)