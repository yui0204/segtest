from keras.layers import merge, MaxPooling2D, UpSampling2D, core

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout
from keras.layers.merge import concatenate


def Unet(n_classes, input_height=256, input_width=512, nChannels=3): 
    
    inputs = Input((input_height, input_width, nChannels))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([up1, conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([up2, conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    
    conv6 = Conv2D(n_classes, (1, 1), activation='relu',padding='same')(conv5)
    #conv6 = core.Reshape((n_Classes, input_height * input_width))(conv6)
    #conv6 = core.Permute((2,1))(conv6)

    if n_classes == 1:
        conv7 = core.Activation('sigmoid')(conv6)
        print("Activation of the final layer is sigmoid")
    else:
        conv7 = core.Activation('softmax')(conv6)
        print("Activation of the final layer is softmax")
    
    model = Model(input=inputs, output=conv7)
	
    return model
	