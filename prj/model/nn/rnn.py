import keras as tfk
from keras import layers as tfkl
from keras import metrics as tfkm
from keras import optimizers as tfko
from prj.model.nn.neural import TabularNNModel


class Rnn(TabularNNModel):    
    def __init__(self,
                 layers:int=1,
                 start_neurons:int=64, 
                 decay_factor:int=1, 
                 momentum=0.6,
                 output_units:int=32, 
                 use_lstm:bool=True, 
                 use_batch_norm:bool=True,
                 dropout:float=0.2,
                 **kwargs):

        self.start_neurons = start_neurons
        self.layers = layers
        self.momentum=momentum
        self.decay_factor = decay_factor
        self.output_units = output_units
        self.use_lstm = use_lstm
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        
        super().__init__(**kwargs)
                         
            
    def _build(self):
        input, x = self._build_input_layers()
        
        
        RNN = tfkl.LSTM if self.use_lstm else tfkl.GRU

        for i in range(self.layers):
            decay = 1 if i == 0 else self.decay_factor * i
            gru_k_units = int(self.start_neurons / decay)
            if gru_k_units < self.output_units:
                break
            
            x = RNN(gru_k_units, return_sequences=True, name=f'rnn_{i+1}')(x)
            if self.use_batch_norm:
                x = tfkl.BatchNormalization(momentum=self.momentum)(x)
            
        gru_output = RNN(self.output_units, name='rnn_output')(x)
        
        x = tfkl.Dropout(self.dropout, seed=self.random_seed)(gru_output)
        
        out = tfkl.Dense(
            self.output_dim, 
            activation=self.output_activation
        )(x)
            
        self.model = tfk.Model(inputs=input, outputs=out, name=self.model_name)

    
    
        
        