from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from keras.layers import Activation, Dropout, Reshape, Permute
from keras.layers import merge, MaxPooling2D, UpSampling2D, core
from keras.applications.vgg16 import VGG16




def FCN32(n_classes, input_height=256, input_width=512, nChannels=3):
    inputs = Input((input_height, input_width, nChannels))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    """
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #1 = x    
    
	# Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #f2 = x

	# Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #f3 = x

	# Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #f4 = x

	# Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)  
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x) 
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    """
    
    inputs2 = Input((vgg16.output_shape[1:]))
    
    x = (Conv2D(4096, (7, 7), activation='relu', padding='same'))(inputs2)
    x = Dropout(0.5)(x)
    x = (Conv2D(4096, (1, 1) , activation='relu' , padding='same'))(x)
    x = Dropout(0.5)(x)
    
    x = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal'))(x)
    x = Conv2DTranspose(n_classes, kernel_size=(32,32), strides=(32,32), 
                        use_bias=False)(x)
    
    if n_classes == 1:
        x = (Activation('sigmoid'))(x)
        print("Activation of the final layer is sigmoid")
    else:
        x = (Activation('softmax'))(x)
        print("Activation of the final layer is softmax")
	
    #model = Model(inputs=inputs, outputs=x)
    
    model = Model(inputs=inputs2, outputs=x)
    model = Model(inputs=vgg16.input, outputs=model(vgg16.output))

    return model
