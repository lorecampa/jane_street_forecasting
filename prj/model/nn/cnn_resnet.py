import keras as tfk
from keras import layers as tfkl


from prj.model.nn.neural import TabularNNModel

class CnnResnet(TabularNNModel):
    ID = 'DeepCNNResnet'
    def __init__(self,
                 conv_filters:int=32, 
                 kernel_size:int=2,
                 kernel_initializer:str='he_uniform',
                 dropout:float=0.2, 
                 rnn_neurons:int=64, 
                 use_lstm:bool=False,
                 rnn_decay_factor:int=2, 
                 layers:int=7,
                 **kwargs):
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.rnn_decay_factor = rnn_decay_factor
        self.kernel_initializer=kernel_initializer
        self.dropout = dropout
        self.rnn_neurons = rnn_neurons
        self.use_lstm = use_lstm
        self.layers=layers
        
        super().__init__(**kwargs)        
    
    def _build_residual_block(self, input_layer, input_dim, num_filters, kernel_size, dilation, block_name):
        # a resnet block that contains two dilated causal convolutions
        conv1 = tfkl.Conv1D(num_filters, kernel_size, padding='causal', name=f'{block_name}_conv1', 
                            dilation_rate=dilation, use_bias=False, kernel_initializer=self.kernel_initializer)(input_layer)
        conv1 = tfkl.BatchNormalization(name=f'{block_name}_norm1')(conv1)
        conv1 = tfkl.Activation('relu')(conv1)
        conv1 = tfkl.Dropout(self.dropout)(conv1)
        
        conv2 = tfkl.Conv1D(num_filters, kernel_size, padding='causal', name=f'{block_name}_conv2', 
                            dilation_rate=dilation, use_bias=False, kernel_initializer=self.kernel_initializer)(conv1)
        conv2 = tfkl.BatchNormalization(name=f'{block_name}_norm2')(conv2)
        conv2 = tfkl.Activation('relu')(conv2)
        conv2 = tfkl.Dropout(self.dropout)(conv2)

        # 1d convolution useful when the input have a different dimension w.r.t. the output of the convolutions
        # since the two blocks need to be added at the end
        if input_dim != num_filters:
            downsample = tfkl.Conv1D(num_filters, 1, name=f'{block_name}_downsample', use_bias=False)(input_layer)
        else:
            downsample = input_layer

        out=tfkl.Add()([conv2, downsample])
        out = tfkl.Activation('relu')(out)
        return out
    
    
    def _build(self):
        input, x = self._build_input_layers()

        time_dim = self.input_dim[0]
        pool_size = 4
        avg_pooling_window = tfkl.AveragePooling1D(pool_size=min(pool_size, time_dim), padding='valid')(x) if pool_size > 1 else x
        res_output=avg_pooling_window
                
        # dilated causal convolutions to increase the receptive field
        for i in range(self.layers):
            res_output = self._build_residual_block(
                input_layer=res_output, 
                input_dim=self.conv_filters if i > 0 else self.input_dim,
                num_filters=self.conv_filters,
                kernel_size=self.kernel_size,
                dilation=2**i,
                block_name=f'conv_full_resnet_block_{i}') 
            
        RNN = tfkl.LSTM if self.use_lstm else tfkl.GRU   
        rnn_full = RNN(self.rnn_neurons, name='rnn_full')(res_output)

        # convolutions using the downsampled series with window 2
        # first_conv = tfkl.Conv1D(self.conv_filters, 1, name='initial_conv_down2', use_bias=False)(input_layer)
        pool_size *= 2
        avg_pooling_window = tfkl.AveragePooling1D(pool_size=min(pool_size, time_dim), padding='valid')(x)
        res_output = avg_pooling_window
        for i in range(self.layers-1):
            res_output = self._build_residual_block(
                input_layer=res_output, 
                input_dim=self.conv_filters if i > 0 else self.input_dim,
                num_filters=self.conv_filters,
                kernel_size=self.kernel_size,
                dilation=2**i,
                block_name=f'conv_down{pool_size}_resnet_block_{i}')
        rnn_window2 = RNN(int(self.rnn_neurons / self.rnn_decay_factor), name=f'rnn_down{pool_size}')(res_output)

        # convolutions using the downsampled series with window 4
        # first_conv = tfkl.Conv1D(self.conv_filters, 1, name='initial_conv_down4', use_bias=False)(input_layer)
        pool_size*=2    
        avg_pooling_window = tfkl.AveragePooling1D(pool_size=min(pool_size, time_dim), padding='valid')(x)
        res_output = avg_pooling_window
        for i in range(self.layers-3):
            res_output = self._build_residual_block(
                input_layer=res_output, 
                input_dim=self.conv_filters if i > 0 else self.input_dim,
                num_filters=self.conv_filters,
                kernel_size=self.kernel_size,
                dilation=2**i,
                block_name=f'conv_down{pool_size}_resnet_block_{i}')
        rnn_window4 = RNN(int(self.rnn_neurons / (self.rnn_decay_factor*2)), name=f'rnn_down{pool_size}')(res_output)
            
        # concatenate the extracted features
        concat = tfkl.Concatenate()([rnn_full, rnn_window2, rnn_window4])
        dropout = tfkl.Dropout(self.dropout, name='Dropout_RNN')(concat)
        
        out = tfkl.Dense(self.output_dim, name='output_rnn', activation=self.output_activation)(dropout)
        
        self.model = tfk.Model(inputs=input, outputs=out, name=self.model_name)