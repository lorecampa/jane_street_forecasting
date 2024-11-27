import keras as tfk
from keras import layers as tfkl
from prj.config import GLOBAL_SEED
from prj.model.nn.neural import TabularNNModel


class MLP(TabularNNModel):
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
        n_layers: int = 1,
        start_units: int = 128,
        units_decay: int = 2,
        dropout_rate: float = 0.1,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-4,
        activation: str = 'relu',
        **kwargs
    ):
        '''
        Args:
            n_layers (int): the number of layers
            start_units (int): the number of hidden units in the first layer
            units_decay (int): the decay to decrease the number of hidden units at each layer
            dropout_rate (float): the dropout rate
            l1_lambda (float): l1 regularization coefficient
            l2_lambda (float): l2 regularization coefficient
            activation (str): the activation function of the hidden layers
        '''
        super(MLP, self).__init__(
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
        self.n_layers = n_layers
        self.start_units = start_units
        self.units_decay = units_decay
        self.dropout_rate = dropout_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.activation = activation
        self.output_dim = output_dim
    
    def _build(self):
        input, x = self._build_input_layers()

        units = self.start_units
        for i in range(self.n_layers):
            x = tfkl.Dense(
                units=units,
                kernel_initializer=tfk.initializers.HeNormal(seed=self.random_seed),
                kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
                name=f'Dense{i}'
            )(x)
            x = tfkl.BatchNormalization(name=f'BatchNormalization{i}')(x)
            x = tfkl.Activation(self.activation, name=f'Activation{i}')(x)
            x = tfkl.Dropout(self.dropout_rate, name=f'Dropout{i}')(x)
            units = int(units / self.units_decay)

        outputs = tfkl.Dense(
            units=self.output_dim,
            kernel_initializer=tfk.initializers.GlorotUniform(seed=self.random_seed),
            kernel_regularizer=tfk.regularizers.l1_l2(l1=self.l1_lambda, l2=self.l2_lambda),
            name='OutputDense',
            activation='linear'
        )(x)
        self.model = tfk.Model(inputs=input, outputs=outputs)