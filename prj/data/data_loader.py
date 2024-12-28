from pathlib import Path
import typing

import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from prj.config import DATA_DIR
import polars as pl
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from prj.utils import wrapper_pbar

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
        self.include_intrastock_norm = kwargs.get('include_intrastock_norm', False)
        self.include_knn_features = kwargs.get('include_knn_features', False)
        self.include_intrastock_norm_temporal = kwargs.get('include_intrastock_norm_temporal', False)
        
            
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
        self.include_symbol_id = config.include_symbol_id
        self.include_time_id = config.include_time_id
        self.include_intrastock_norm = config.include_intrastock_norm
        self.include_knn_features = config.include_knn_features
        self.include_intrastock_norm_temporal = config.include_intrastock_norm_temporal
        
        self.categorical_features = ['feature_09', 'feature_10', 'feature_11']
        # self.categorical_features = []
        self.target = "responder_6"
        self.features = None
        self.window_period = 7                    
        
    def _build_splits(self, df: pl.LazyFrame):
        X = df.select(self.features).cast(pl.Float32).collect().to_numpy()
        y = df.select(self.target).cast(pl.Float32).collect().to_series().to_numpy()
        w = df.select('weight').cast(pl.Float32).collect().to_series().to_numpy()
        info = df.select(['date_id', 'time_id', 'symbol_id']).collect().to_numpy()
        return X, y, w, info
                
    def load_train_and_val(self, start_dt: int, end_dt = None, val_ratio: float = 0.1):
        end_dt = end_dt if end_dt is not None else PARTITIONS_DATE_INFO[9]['max_date']
        
        assert val_ratio >= 0 and val_ratio <= 1, 'val_ratio must be in (0, 1)'
        df = self.load(start_dt, end_dt)
        
        dates = df.select('date_id').unique().collect().to_series().sort()
        split_point = int(len(dates) * (1 - val_ratio))
        split_date = dates[split_point] if val_ratio > 0 else dates[-1] + 1

        
        df_train = df.filter(pl.col('date_id').lt(split_date))
        df_val = df.filter(pl.col('date_id').ge(split_date))
        
        return df_train, df_val
    
    def load_numpy_with_partition(self, start_part_id: int, end_part_id: int = None):
        start_dt = PARTITIONS_DATE_INFO[start_part_id]['min_date']
        end_dt = PARTITIONS_DATE_INFO[end_part_id]['max_date'] if end_part_id is not None else PARTITIONS_DATE_INFO[9]['max_date']
        return self.load_numpy(start_dt, end_dt)
        
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
        assert end_dt is None or start_dt <= end_dt, 'start_dt must be less than or equal end_dt'
        
        end_dt = end_dt if end_dt is not None else PARTITIONS_DATE_INFO[9]['max_date']
        df = self._load(max(0, start_dt - self.window_period), end_dt).filter(
            pl.col('date_id').is_between(start_dt, end_dt, closed='both'),
        )
                
        return df
    
    def _load(self, start_dt, end_dt) -> pl.LazyFrame:
        data_path = Path(self.data_dir)
        if self.ffill:
            data_path = data_path / 'train_ffill.parquet'
        else:
            data_path = data_path / 'train.parquet'
            
        df = pl.scan_parquet(
            data_path
        ).filter(pl.col('date_id').is_between(start_dt, end_dt))\
        .sort('date_id', 'time_id')
            
        self.features = [f'feature_{i:02d}' for i in range(79)]
        if self.include_symbol_id:
            self.features.append('symbol_id')
        if self.include_time_id:
            df = df.with_columns(
                pl.col('time_id').truediv(pl.col('time_id').max().over('date_id', 'symbol_id')).alias('time_id_norm')
            )
            self.features.append('time_id_norm')
                
        if self.include_lags:
            lags = self._compute_lags(df)
            df = self._include_lags(df, lags)
            
        if self.include_intrastock_norm:
            df = self._include_intrastock_norm(df)
            
        if self.include_knn_features:
            df = self._include_knn_features(df)
            
        if self.include_intrastock_norm_temporal:
            df = self._include_intrastock_norm_temporal(df)    
                
        df = self._impute(df)
                        
        return df
    
    def _impute(self, df: pl.LazyFrame) -> pl.LazyFrame:
        impute_cols = [col for col in self.features if col not in ['symbol_id', 'time_id', 'date_id']]
        # if self.ffill:
        #     df = df.with_columns(
        #         pl.col(impute_cols).fill_nan(None).fill_null(strategy="forward", limit=10).over('symbol_id')
        #     )
        if self.zero_fill:
            df = df.with_columns(
                pl.col(impute_cols).fill_nan(None).fill_null(strategy="zero")
            )
        return df
    
    
    def _include_intrastock_norm(self, df: pl.LazyFrame) -> pl.LazyFrame:
        MEAN_FEATURES = [0, 2, 3, 5, 6, 7, 18, 19, 34, 35, 36, 37, 38, 41, 43, 44, 48, 53, 55, 59, 62, 65, 68, 73, 74, 75, 76, 77, 78]
        STD_FEATURES = [39, 42, 46, 53, 57, 66]
        SKEW_FEATURES = [5, 40, 41, 42, 43, 44]
        ZSCORE_FEATURES = [1, 36, 40, 45, 48, 49, 51, 52, 53, 54, 55, 59, 60]
        
        df = df.with_columns(
            pl.col([f'feature_{j:02d}' for j in set(MEAN_FEATURES + ZSCORE_FEATURES)]).mean().over(['date_id', 'time_id']).name.suffix('_mean'),
            pl.col([f'feature_{j:02d}' for j in set(STD_FEATURES + ZSCORE_FEATURES)]).std().over(['date_id', 'time_id']).name.suffix('_std'),
            pl.col([f'feature_{j:02d}' for j in SKEW_FEATURES]).skew().over(['date_id', 'time_id']).name.suffix('_skew'),
        ).with_columns(
            pl.col(f'feature_{j:02d}').sub(f'feature_{j:02d}_mean').truediv(f'feature_{j:02d}_std').name.suffix('_zscore') for j in ZSCORE_FEATURES
        ).drop([f'feature_{j:02d}_std' for j in ZSCORE_FEATURES if j not in STD_FEATURES] + \
            [f'feature_{j:02d}_mean' for j in ZSCORE_FEATURES if j not in MEAN_FEATURES])

        intrastock_features = [f'feature_{j:02d}_mean' for j in MEAN_FEATURES] + \
            [f'feature_{j:02d}_std' for j in STD_FEATURES] + [f'feature_{j:02d}_skew' for j in SKEW_FEATURES] + \
            [f'feature_{j:02d}_zscore' for j in ZSCORE_FEATURES]
        self.features.extend(intrastock_features)
        
        return df
    
    def _compute_lags(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.select(
            'date_id', 'symbol_id', *[f"responder_{idx}" for idx in range(9)]
        ).with_columns(
            pl.col('date_id').add(1),
        ).rename(
            {f"responder_{idx}": f"responder_{idx}_lag_1" for idx in range(9)}
        ).group_by(['date_id', 'symbol_id'], maintain_order=True)\
        .last()
    
    def _include_lags(self, df: pl.LazyFrame, lags: pl.LazyFrame) -> pl.LazyFrame:        
        df = df.join(lags, on=['date_id', 'symbol_id'], how='left', maintain_order='left')
        self.features.extend([f"responder_{idx}_lag_1" for idx in range(9)])
        return df
    
    
    def _include_knn_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        def _fn(df: pl.DataFrame, 
                         period: int,
                         schema: dict,
                         features: list[str],
                         target: str,
                         knn_windows: list[int],
                         aggs: list[str]) -> pl.DataFrame:
    
            min_date = df['date_id'].min()
            min_time = df['time_id'].min()
            max_date = df['date_id'].max()
            max_time = df['time_id'].max()
            # symbol_id = df['symbol_id'].unique().item()
            # print(f"{df.shape}, [{min_date} - {max_date}] [{min_time} - {max_time}] , days={(max_date - min_date)}, period={period}, symbol_id={symbol_id}")

            base_cols = ['date_id', 'time_id', 'symbol_id', target]
            if (max_date - min_date) != period:
                print(f"{df.shape}, [{min_date} - {max_date}] [{min_time} - {max_time}] , days={(max_date - min_date)}, period={period}")
                return pl.DataFrame({}, schema=schema)
            
            df = df.fill_nan(None).drop_nulls()
            
            symbols = df['symbol_id'].unique().to_numpy()
            final_res = pl.DataFrame({}, schema)
            for symbol_id in symbols:
                target_samples = df.filter(
                    pl.col('date_id') == max_date,
                    pl.col('symbol_id') == symbol_id,
                ).select(features).to_numpy()
                
                res = df.filter(
                    pl.col('date_id') == max_date,
                    pl.col('symbol_id') == symbol_id
                ).select(base_cols)
                
                samples = df.filter(
                    pl.col('date_id') < max_date,
                    pl.col('symbol_id') == symbol_id,
                ).select(features + [target]).to_numpy()
                
                if target_samples.shape[0] == 0 or samples.shape[0] == 0:
                    print(f"No samples for symbol_id={symbol_id} {min_date}-{max_date}")
                    continue
                    
                distances = pairwise_distances(target_samples, samples[:, :-1])
                sorted_indices = np.argsort(distances, axis=-1)
                for n in knn_windows:
                    if n > distances.shape[1]:
                        print(f'Not enough samples to compute knn with window {n} and distances shape {distances.shape}')
                        continue # not enough samples to compute knn with that window

                    nearest_indices = sorted_indices[:, :n]        
                    knn_targets: np.ndarray = samples[nearest_indices.flatten(), -1].reshape(target_samples.shape[0], n)
                    col_name = f'{target}_knn_{n}'
                    for agg in aggs:
                        agg_name = f'{col_name}_{agg}'
                        if agg == 'mean':
                            res = res.with_columns(
                                pl.Series(np.mean(knn_targets, axis=-1)).alias(agg_name),
                            )
                        elif agg == 'weighted_mean':
                            knn_distances = distances[np.arange(distances.shape[0])[:, None], nearest_indices]
                            weights = 1 / (knn_distances + 1e-8)
                            weights /= np.sum(weights, axis=-1, keepdims=True)
                            res = res.with_columns(
                                pl.Series(np.mean(knn_targets * weights, axis=-1)).alias(agg_name)
                            )
                        elif agg == 'positive_ratio':
                            res = res.with_columns(
                                pl.Series(np.mean(knn_targets > 0, axis=-1)).alias(agg_name)
                            )
                        else:
                            raise ValueError(f'Unknown aggregation {agg}')
                final_res = final_res.vstack(res.select(*list(schema.keys())).cast(schema))
                            
            return final_res.select(*list(schema.keys())).cast(schema)
        knn_windows = [5, 10]
        aggs = ['mean', 'weighted_mean', 'positive_ratio']
        
        n_dates = df.select('date_id').unique().count().collect().item()
        num_groups = max(0, n_dates - self.window_period + 1)
        schema = {'date_id': pl.Int16, 'time_id': pl.Int16, 'symbol_id': pl.Int8, self.target: pl.Float32}
        
        schema.update({
            f'{self.target}_knn_{n}_{agg}': pl.Float32
            for n in knn_windows
            for agg in aggs
        })

        with tqdm(total=int(num_groups)) as pbar:
            knn_df = df.sort('date_id', 'time_id').group_by_dynamic(
                pl.col('date_id').cast(pl.Int64),
                period=f"{self.window_period}i",
                every="1i",
                closed='both',
            ).map_groups(
                wrapper_pbar(
                    pbar, 
                    lambda x: _fn(x, self.window_period, schema, self.features, self.target, knn_windows, aggs)
                ),
                schema=schema
            ).drop(self.target)\
            .collect()
        
        df = df.join(knn_df, on=['date_id', 'time_id', 'symbol_id'], how='left')
        return df
    
    
    def _include_intrastock_norm_temporal(self, df: pl.LazyFrame) -> pl.LazyFrame:
        DEFAULT_CLUSTER = -1
        
        def _fn(batch: pl.DataFrame, period, responders, schema) -> pl.DataFrame:
            min_date = batch['date_id'].min()
            max_date = batch['date_id'].max()
            # print(f"Processing {min_date}-{max_date}")
            
            if max_date - min_date != period:
                return pl.DataFrame({}, schema=schema)
            
            pivot = batch.filter(pl.col('date_id') < max_date)\
                .pivot(index=['date_id', 'time_id'], values=responders, separator='_', on='symbol_id')\
                .fill_nan(None)\
                .fill_null(strategy='zero')
                
            res = batch.select('date_id', 'time_id', 'symbol_id')\
                .filter(pl.col('date_id') == max_date)
            
            for responder in responders:
                cols = [col for col in pivot.columns if col not in ['date_id', 'time_id']]
                stocks = [int(col) for col in cols]
                df_corr_responder = pivot.select(cols).corr()
                linked = linkage(df_corr_responder, method='ward')
                cluster_labels = fcluster(linked, t=2.5, criterion='distance')
                
                
                # print(stocks, cluster_labels)
                res = res.with_columns(
                    pl.col('symbol_id').replace_strict(
                        old=stocks, new=cluster_labels, default=DEFAULT_CLUSTER, return_dtype=pl.Int8
                    ).alias(f'cluster_label_{responder}')
                )
            
            return res.cast(schema)


        responders = ['responder_6']
        schema = {'date_id': pl.Int16, 'time_id': pl.Int16, 'symbol_id': pl.Int8}
        for responder in responders:
            schema[f'cluster_label_{responder}'] = pl.Int8
            
                        
        n_days = df.select('date_id').collect().n_unique()
        with tqdm(total=int(n_days)) as pbar:
            clusters = df.select('date_id', 'time_id', 'symbol_id', *responders).sort('date_id').group_by_dynamic(
                pl.col('date_id').cast(pl.Int64),
                period=f'{self.window_period}i',
                every='1i',
                closed='both',
            ).map_groups(
                wrapper_pbar(
                    pbar, 
                    lambda x: _fn(x, period=self.window_period, responders=responders, schema=schema),
                ),
                schema=schema
            ).collect()
        
        cluster_label_cols = [col for col in clusters.columns if col.startswith('cluster_label')]
        df = df.join(
            clusters.lazy(), on=['date_id', 'time_id', 'symbol_id'], how='left'
        ).with_columns(
            pl.col(cluster_label_cols).fill_nan(None).fill_null(DEFAULT_CLUSTER)
        )
        
        MEAN_FEATURES = [0, 2, 3, 5, 6, 7, 18, 19, 34, 35, 36, 37, 38, 41, 43, 44, 48, 53, 55, 59, 62, 65, 68, 73, 74, 75, 76, 77, 78]
        STD_FEATURES = [39, 42, 46, 53, 57, 66]
        SKEW_FEATURES = [5, 40, 41, 42, 43, 44]
        ZSCORE_FEATURES = [1, 36, 40, 45, 48, 49, 51, 52, 53, 54, 55, 59, 60]
        def _include_intrastock_norm(_df: pl.LazyFrame, responder) -> pl.LazyFrame:
            _df = _df.with_columns(
                pl.col([f'feature_{j:02d}' for j in set(MEAN_FEATURES + ZSCORE_FEATURES)]).mean().over(['date_id', 'time_id', f'cluster_label_{responder}']).name.suffix(f'_{responder}_mean'),
                pl.col([f'feature_{j:02d}' for j in set(STD_FEATURES + ZSCORE_FEATURES)]).std().over(['date_id', 'time_id', f'cluster_label_{responder}']).name.suffix(f'_{responder}_std'),
                pl.col([f'feature_{j:02d}' for j in SKEW_FEATURES]).skew().over(['date_id', 'time_id', f'cluster_label_{responder}']).name.suffix(f'_{responder}_skew'),
            ).with_columns(
                pl.col(f'feature_{j:02d}').sub(f'feature_{j:02d}_{responder}_mean').truediv(f'feature_{j:02d}_{responder}_std').name.suffix(f'_{responder}_zscore') for j in ZSCORE_FEATURES
            ).drop([f'feature_{j:02d}_{responder}_std' for j in ZSCORE_FEATURES if j not in STD_FEATURES] + \
                [f'feature_{j:02d}_{responder}_mean' for j in ZSCORE_FEATURES if j not in MEAN_FEATURES])
            
            intrastock_features = [f'feature_{j:02d}_{responder}_mean' for j in MEAN_FEATURES] + \
                [f'feature_{j:02d}_{responder}_std' for j in STD_FEATURES] + [f'feature_{j:02d}_{responder}_skew' for j in SKEW_FEATURES] + \
                [f'feature_{j:02d}_{responder}_zscore' for j in ZSCORE_FEATURES]
            
            self.features.extend(intrastock_features)
            return _df
            
        
        for responder in responders:
            df = _include_intrastock_norm(df, responder)
        
        df = df.drop(cluster_label_cols)
        
        return df
    
        
        
        

    def get_info(self) -> None:
        return {
            **self.config.__dict__,
            'features': self.features,
        }
    
    
    
    