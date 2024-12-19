from pathlib import Path
import typing
from prj.config import DATA_DIR
import polars as pl

PARTITIONS_DATE_INFO = {
    0: {'min_date': 0, 'max_date': 169},
    1: {'min_date': 170, 'max_date': 339},
    2: {'min_date': 340, 'max_date': 509},
    3: {'min_date': 510, 'max_date': 679},
    4: {'min_date': 680, 'max_date': 849},
    5: {'min_date': 850, 'max_date': 1019},
    6: {'min_date': 1020, 'max_date': 1189},
    7: {'min_date': 1190, 'max_date': 1359},
    8: {'min_date': 1360, 'max_date': 1529},
    9: {'min_date': 1530, 'max_date': 1698}
}

class DataConfig:
    def __init__(self, **kwargs):
        self.ffill = kwargs.get('ffill', False)
        self.zero_fill = kwargs.get('zero_fill', False)
        self.include_lags = kwargs.get('include_lags', False)
        self.include_symbol_id = kwargs.get('include_symbol_id', False)
        self.include_time_id = kwargs.get('include_time_id', False)
        
            
class DataLoader:
    def __init__(
        self, 
        data_dir: typing.Union[str | Path],
        config: DataConfig = DataConfig(),
    ):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
            
        self.data_dir = data_dir
        self.config = config
        
        
        self.ffill = config.ffill
        self.zero_fill = config.zero_fill
        
        self.include_lags = config.include_lags
        self.include_symbol_id = config.include_lags
        self.include_time_id = config.include_time_id
        
        self.target = "responder_6"
        
        self.features = [f'feature_{i:02d}' for i in range(79)]
        if self.include_symbol_id:
            self.features.append('symbol_id')
        if self.include_time_id:
            self.features.append('time_id')
        if self.include_lags:
            self.features.extend([f"responder_{idx}_lag_1" for idx in range(9)])
            
        self.time_cols = ['date_id', 'time_id']
        
    
    def _build_splits(self, df: pl.LazyFrame):
        X = df.select(self.features).cast(pl.Float32).collect().to_numpy()
        y = df.select(self.target).cast(pl.Float32).collect().to_series().to_numpy()
        w = df.select('weight').cast(pl.Float32).collect().to_series().to_numpy()
        info = df.select(self.time_cols + ['symbol_id']).collect().to_numpy()
        return X, y, w, info
    
    def load_train_and_val(self, start_dt: int, end_dt: None, val_ratio: float):
        assert val_ratio >= 0 and val_ratio <= 1, 'val_ratio must be in (0, 1)'
        df = self.load(start_dt, end_dt)
        
        dates = df.select('date_id').unique().collect().to_series().sort()
        split_point = int(len(dates) * (1 - val_ratio))
        split_date = dates[split_point] if val_ratio > 0 else dates[-1] + 1

        
        df_train = df.filter(pl.col('date_id').lt(split_date))
        df_val = df.filter(pl.col('date_id').ge(split_date))
        
        return df_train, df_val
        
        
    def load_numpy(self, start_dt: int, end_dt: int = None):
        df = self.load(start_dt, end_dt)
        return self._build_splits(df)
    
    def load_with_partition(self, start_part_id: int, end_part_id: int = None):
        start_dt = PARTITIONS_DATE_INFO[start_part_id]['min_date']
        end_dt = PARTITIONS_DATE_INFO[end_part_id]['max_date'] if end_part_id is not None else PARTITIONS_DATE_INFO[9]['max_date']
        return self.load(start_dt, end_dt)
        
    def load(self, start_dt: int, end_dt: int = None) -> pl.LazyFrame:
        assert start_dt >= PARTITIONS_DATE_INFO[0]['min_date'] and start_dt <= PARTITIONS_DATE_INFO[9]['max_date'], 'start_dt out of range'
        assert end_dt is None or (end_dt >= PARTITIONS_DATE_INFO[0]['min_date'] and end_dt <= PARTITIONS_DATE_INFO[9]['max_date']), 'end_dt out of range'
        assert start_dt <= end_dt, 'start_dt must be less than or equal end_dt'
        
        end_dt = end_dt if end_dt is not None else PARTITIONS_DATE_INFO[9]['max_date']
        df = self._load().filter(
            pl.col('date_id').is_between(start_dt, end_dt, closed='both'),
        )
                
        if self.ffill:
            df = df.with_columns(
                pl.col(self.features).fill_nan(None).fill_null(strategy="forward", limit=10).over('symbol_id')
            )
        if self.zero_fill:
            df = df.with_columns(
                pl.col(self.features).fill_nan(None).fill_null(strategy="zero")
            )
        return df
    
    def _load(self) -> pl.LazyFrame:
        df = pl.scan_parquet(
            self.data_dir
        )
        # preprocessing
        if self.include_lags:
            lags = df.select(
                'date_id', 'symbol_id', *[f"responder_{idx}" for idx in range(9)]
            ).with_columns(
                pl.col('date_id').add(1),
            ).rename(
                {f"responder_{idx}": f"responder_{idx}_lag_1" for idx in range(9)}
            ).group_by(['date_id', 'symbol_id'], maintain_order=True)\
            .last()
            df = df.join(lags, on=['date_id', 'symbol_id'], how='left', maintain_order='left')
                    
        
        return df

    def get_info(self) -> None:
        return {
            **self.config.__dict__,
            'features': self.features,
        }
    
    
    
    