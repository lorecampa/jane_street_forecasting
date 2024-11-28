import typing
import warnings
import keras as tfk
from keras import layers as tfkl
import os
from typing_extensions import Union, Tuple
import polars as pl
from abc import ABC, abstractmethod
import joblib
import numpy as np
from prj.config import GLOBAL_SEED


class TabularNNModel(ABC):
    '''
    The base class for each tabular neural network model.
    
    This class handle the null values by just filling them with zero. To better handle them
    treat them separately outside the class.
    '''
    
    def __init__(
        self,
        input_dim: tuple,
        output_dim: int = 1,
        use_gaussian_noise: bool = False,
        gaussian_noise_std: float = 0.01,
        verbose: bool = False, 
        model_name: str = 'model',
        random_seed: int = GLOBAL_SEED,
        logger = print,
        **kwargs
    ):
        '''
        Args:
            categorical_features (List[str]): the list of categorical features
            numerical_features (List[str]): the list of numerical features
            categorical_transform (str): the type of categorical encoder, can be one-hot-encoding, target-encoding
                or embeddings (in this case the categorical variables will go through a learnable embedding layer)
            numerical_transform (str): the type of numerical preprocessing. Can be any between: "yeo-johnson", "standard", 
                "quantile-normal", "quantile-normal", "max-abs". If None, no preprocessing is done
            use_gaussian_noise (bool): if True, applies a gaussian noise to the input
            gaussian_noise_std (float): the standard deviation of the gaussian noise
            max_categorical_embedding_dim (int): the maximum size of a categorical embedding. The actual size will be
                computed as min(max_categorical_embedding_dim, (vocabulary_size + 1) // 2) for each category separately
            verbose (bool)
            model_name (str)
            random_seed (int)
        '''
        super(TabularNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.verbose = verbose
        self.use_gaussian_noise = use_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        self.random_seed = random_seed
        self.model_name = model_name
        self.logger = logger
        self.model: tfk.Model = None

    def __call__(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        sample_weight=None,
        validation_data: typing.Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        early_stopping_rounds: int = 1,
        batch_size: int = 256,
        epochs: int = 1,
        optimizer: tfk.optimizers.Optimizer = None, 
        loss: tfk.losses.Loss = None, 
        metrics: typing.List[Union[str, tfk.metrics.Metric]] = [],
        early_stopping_monitor: str = 'val_loss',
        early_stopping_mode: str = 'auto',
        lr_scheduler: Union[callable, tfk.callbacks.Callback] = None,
        save_checkpoints: bool = True,
        checkpoint_dir: str = None,
    ):
        
        callbacks = []     
        if validation_data is not None:
            self.logger(f'Training with early stopping patience {early_stopping_rounds}')
            early_stopping = tfk.callbacks.EarlyStopping(
                monitor=early_stopping_monitor, 
                patience=early_stopping_rounds, 
                mode=early_stopping_mode, 
                restore_best_weights=True
            )
            callbacks.append([early_stopping])
        
            
        if lr_scheduler is not None:
            if type(lr_scheduler) == callable:
                scheduler = tfk.callbacks.LearningRateScheduler(lr_scheduler)
                callbacks.append(scheduler)
            else:
                callbacks.append(lr_scheduler)
                
        callbacks.append(tfk.callbacks.TerminateOnNaN())
        
        if self.model is not None:
            warnings.warn('Model already compiled. Recompiling and training')
        
        self._build()
        compile_args = dict(
            loss=loss,
            optimizer=optimizer,
        )
        if sample_weight is None:
            compile_args.update(dict(
                metrics=metrics
            ))
        else:
            compile_args.update(dict(
                weighted_metrics=metrics
            ))
        self.model.compile(**compile_args)
        self.model.summary(print_fn=self.logger)
        
        if save_checkpoints and checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.save(checkpoint_dir, with_model=False)
            self.logger(f'Checkpoints will be saved at {checkpoint_dir}')
            callbacks.append(tfk.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'checkpoint.weights.h5'),
                save_weights_only=True,
                monitor=early_stopping_monitor if validation_data else 'loss',
                mode=early_stopping_mode if validation_data else 'auto',
                save_best_only=True))
        
        fit_history = self.model.fit(
            X,
            y,
            batch_size=batch_size,
            epochs=epochs,
            sample_weight=sample_weight,
            validation_data=validation_data,
            validation_batch_size=batch_size if validation_data is not None else None,
            callbacks=callbacks
        ).history
        
        self.logger(f'Fit complete after {len(fit_history["loss"])}')
        
        if save_checkpoints and checkpoint_dir:
            self.model.load_weights(os.path.join(checkpoint_dir, 'checkpoint.weights.h5'))
            pl.DataFrame(fit_history).write_csv(os.path.join(checkpoint_dir, 'history.csv'))
        
    def predict(self, X, batch_size=256, **kwargs):
        pred = self.model.predict(X, batch_size=batch_size, **kwargs)
        return pred.flatten() if pred.shape[1] == 1 else pred
        
    def summary(self, expand_nested=True, **kwargs):
        self.model.summary(expand_nested=expand_nested, **kwargs)
    
    def plot(self, dpi:int=50, expand_nested:bool=True, show_shapes:bool=True):
        assert self.model is not None, 'Model not compiled yet'
        return tfk.utils.plot_model(self.model, expand_nested=expand_nested, show_shapes=show_shapes, dpi=dpi)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'model.keras'))        
        _model = self.model
        self.model = None
        joblib.dump(self, os.path.join(path, 'class.joblib'))
        self.model = _model


    @staticmethod
    def load(path: str) -> 'TabularNNModel':
        _class = joblib.load(os.path.join(path, 'class.joblib'))
        _class.model = tfk.models.load_model(os.path.join(path, 'model.keras'))
        return _class
        
    @abstractmethod
    def _build(self):
        raise NotImplementedError('Method build not implemented')

                
    def _build_input_layers(self):
        input = tfkl.Input(shape=self.input_dim)
        x = input
        if self.use_gaussian_noise:
            x = tfkl.GaussianNoise(self.gaussian_noise_std)(x)
        return input, x