from pathlib import Path
import typing
from prj.config import DATA_DIR
import polars as pl

class DataLoader:
    def __init__(
        self, 
        data_dir: typing.Union[str | Path] = DATA_DIR,
        **kwargs,
    ):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.ffill = kwargs.get('ffill', True)
        
        
    
    def _get_partition_path(self, partition_id):
        path = self.data_dir / f'partition_id={partition_id}'
        if self.ffill:
            path = path / 'part-0_ffill.parquet'
        else:
            path = path / 'part-0.parquet'
        return path
    
    def load_partitions(self, start_partition, end_partition):  
        df = pl.concat([
            pl.scan_parquet(self._get_partition_path(i))
            for i in range(start_partition, end_partition + 1)
        ]).sort('date_id', 'time_id', 'symbol_id')
        
        features = [col for col in list(df.collect_schema().keys()) if col.startswith('feature_')]
        time_cols = ['date_id', 'time_id']
        target_feature = 'responder_6'
        
        df = df.select(time_cols + ['symbol_id', 'weight'] + features + [target_feature])
        X = df.select(features + ['symbol_id']).cast(pl.Float32).collect().to_numpy()
        y = df.select(target_feature).cast(pl.Float32).collect().to_series().to_numpy()
        w = df.select('weight').cast(pl.Float32).collect().to_series().to_numpy()
        
        return X, y, w
    
    
    
    