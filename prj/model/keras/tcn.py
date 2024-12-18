import keras as tfk
from keras import layers as tfkl
from prj.model.keras.neural import TabularNNModel

class Tcn(TabularNNModel):
    def __init__(self,
                 conv_filters:int=32, 
                 kernel_size:int=2,
                 use_return:bool=True,
                 dropout:float=0.2,
                 layers:int=7,
                 activation:str='relu',
                 kernel_initializer:str='he_uniform',
                 **kwargs):
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.use_return = use_return
        self.dropout = dropout
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.layers = layers
        
        super().__init__(**kwargs)
        
        
    
    def _build_residual_block(self, input_layer, input_dim, num_filters, kernel_size, dilation, block_name):
        # a resnet block that contains two dilated causal convolutions
        conv1 = tfkl.Conv1D(num_filters, kernel_size, padding='causal', name=f'{block_name}_conv1', 
                            dilation_rate=dilation, use_bias=False, kernel_initializer=self.kernel_initializer)(input_layer)
        conv1 = tfkl.BatchNormalization(name=f'{block_name}_norm1')(conv1)
        conv1 = tfkl.Activation(self.activation)(conv1)
        conv1 = tfkl.Dropout(self.dropout)(conv1)
        
        conv2 = tfkl.Conv1D(num_filters, kernel_size, padding='causal', name=f'{block_name}_conv2', 
                            dilation_rate=dilation, use_bias=False, kernel_initializer=self.kernel_initializer)(conv1)
        conv2 = tfkl.BatchNormalization(name=f'{block_name}_norm2')(conv2)
        conv2 = tfkl.Activation(self.activation)(conv2)
        conv2 = tfkl.Dropout(self.dropout)(conv2)

        # 1d convolution useful when the input have a different dimension w.r.t. the output of the convolutions
        # since the two blocks need to be added at the end
        if input_dim != num_filters:
            downsample = tfkl.Conv1D(num_filters, 1, name=f'{block_name}_downsample', use_bias=False)(input_layer)
        else:
            downsample = input_layer

        out=tfkl.Add()([conv2, downsample])
        out = tfkl.Activation(self.activation)(out)
        return out
    
    
    def _build(self):        
        input, x = self._build_input_layers()
        
        encoder_output = x 
        for i in range(self.layers):
            encoder_output = self._build_residual_block(
                input_layer=encoder_output, 
                input_dim=self.conv_filters if i > 0 else self.input_dim, 
                num_filters=self.conv_filters, 
                kernel_size=self.kernel_size, 
                dilation=2**i, 
                block_name=f'res_block_{i + 1}'
            )
        
        x = tfkl.Lambda(lambda x: x[:,-1,:])(encoder_output)
        out = tfkl.Dense(self.output_dim, name='output_rnn', activation=self.output_activation)(x)
        
        self.model = tfk.Model(inputs=input, outputs=out, name="TCN")
