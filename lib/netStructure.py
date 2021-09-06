from tensorflow.keras.layers import Activation, Reshape, Concatenate, BatchNormalization, Conv2D, Input, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model


def conv_relu_2(x_in, n_filters):
    x=Conv2D(filters=n_filters, 
             kernel_size=(3, 3),
             strides=(1,1),
             padding='same',
             data_format='channels_last')(x_in)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Conv2D(filters=n_filters, 
             kernel_size=(3, 3),
             strides=(1,1),
             padding='same',
             data_format='channels_last')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    return x


def net(input_shape):
    d_input = Input(shape=input_shape)

    x01=Conv2D(filters=8, 
             kernel_size=(3, 3),
             strides=(1,1),
             padding='same',
             data_format='channels_last')(d_input)  
    #x01=BatchNormalization()(x01)
    x01=Activation('relu')(x01)

    x11=MaxPooling2D(pool_size=(2, 2))(x01)   
    x11=conv_relu_2(x11, n_filters=16)
      
    x21=MaxPooling2D(pool_size=(2, 2))(x11)  
    x21=conv_relu_2(x21, n_filters=25)
      
    x31=MaxPooling2D(pool_size=(2, 2))(x21) 
    x22=conv_relu_2(x31, n_filters=25)
    x22=UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='nearest')(x22)
    
    x2=Concatenate(axis=-1)([x21, x22])
  
    x12=conv_relu_2(x2, n_filters=25)   
    x12=UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='nearest')(x12)
    
    x1=Concatenate(axis=-1)([x11, x12])
    
    x02=conv_relu_2(x1, n_filters=25)  
    x02=UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='nearest')(x02)
            
    x0=Concatenate(axis=-1)([x01, x02])
     
    x=Conv2D(filters=12, 
             kernel_size=(3, 3),
             strides=(1,1),
             padding='same',
             data_format='channels_last')(x0)
    #x=BatchNormalization()(x)
    x=Activation('relu')(x)

    x=Conv2D(filters=2, 
             kernel_size=(3, 3),
             strides=(1,1),
             padding='same',
             data_format='channels_last')(x)  

    x=Reshape((-1, 2))(x)
    pred=Activation('softmax')(x)
    model=Model(d_input, pred)
   
    return model

