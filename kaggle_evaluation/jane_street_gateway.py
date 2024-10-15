"""Gateway notebook for https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting"""

import os

import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.templates
import polars as pl


class JSGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str, str] | None = None):
        super().__init__(data_paths, file_share_dir=None)
        self.data_paths = data_paths
        self.set_response_timeout_seconds(60)

    def unpack_data_paths(self):
        if not self.data_paths:
            self.test_path = (
                "/kaggle/input/jane-street-realtime-marketdata-forecasting/test.parquet"
            )
            self.lags_path = (
                "/kaggle/input/jane-street-realtime-marketdata-forecasting/lags.parquet"
            )
        else:
            self.test_path, self.lags_path = self.data_paths

    def generate_data_batches(self):
        date_ids = sorted(
            pl.scan_parquet(self.test_path)
            .select(pl.col("date_id").unique())
            .collect()
            .get_column("date_id")
        )
        assert date_ids[0] == 0

        for date_id in date_ids:
            test_batches = pl.read_parquet(
                os.path.join(self.test_path, f"date_id={date_id}"),
            ).group_by("time_id", maintain_order=True)

            lags = pl.read_parquet(
                os.path.join(self.lags_path, f"date_id={date_id}"),
            )

            for time_id, test in test_batches:
                test_data = (test, lags if time_id == 0 else None)
                validation_data = test.select('row_id')
                yield test_data, validation_data


if __name__ == "__main__":
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        gateway = JSGateway()
        # Relies on valid default data paths
        gateway.run()
    else:
        print("Skipping run for now")
