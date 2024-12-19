import keras as tfk
from keras import layers as tfkl
from prj.config import GLOBAL_SEED
from prj.model.keras.neural import TabularNNModel


class Mlp(TabularNNModel):
    '''
    A simple Multi Layer Perceptron, with Dropout and Batch Normalization.
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
        hidden_units: list = [128, 64, 32],
        dropout_rate: float = 0.1,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-4,
        activation: str = 'relu',
        use_dropout: bool = False,
        use_batch_norm: bool = False,
        use_tanh: bool = False,
        **kwargs
    ):
        super(Mlp, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            use_gaussian_noise=use_gaussian_noise,
            gaussian_noise_std=gaussian_noise_std,
            verbose=verbose,
            model_name=model_name,
            random_seed=random_seed,
            logger=logger,
            **kwargs
        )
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.activation = activation
        self.output_dim = output_dim
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.use_tanh=use_tanh
    
    def _build(self):
        input, x = self._build_input_layers()

        for i, units in enumerate(self.hidden_units):            
            x = tfkl.Dense(
                units=units,
                kernel_initializer=tfk.initializers.HeNormal(),
                kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                name=f'dense_{i}'
            )(x)
            if self.use_batch_norm:
                x = tfkl.BatchNormalization()(x)
            x = tfkl.Activation(self.activation)(x)
            if self.use_dropout:
                x = tfkl.Dropout(self.dropout_rate)(x)

        outputs = tfkl.Dense(
            units=self.output_dim,
            kernel_initializer=tfk.initializers.GlorotUniform(),
            kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
            activation='linear'
        )(x)
        
        if self.use_tanh:
            outputs = tfkl.Activation('tanh')(outputs)
        
        self.model = tfk.Model(inputs=input, outputs=outputs)