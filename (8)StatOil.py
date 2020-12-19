# pip freeze

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.



# This kernel is specifically is for Beginners who wants to experiment building CNN using Keras.
# By using this kernel, you can expect to get good score and also learn keras.
# Keras is simple frameworks where we can initialize the model and keep stacking the layers we want.
# It makes building deep neural networks very easy.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
plt.rcParams['figure.figsize'] = 10, 10
# %matplotlib inline


#Load the data.
train = pd.read_json("./statoil-iceberg-classifier-challenge/train.json/data/processed/train.json")
test = pd.read_json("./statoil-iceberg-classifier-challenge/test.json/data/processed/test.json")


#%%
# Intro about the Data.
# Sentinet -1 sat is at about Km above earth.
# Sending pulses of signals at a particular angle of incidence and then recoding it back.
# Basically those reflected signals are called backscatter.
# The data we have been given is backscatter coefficient which is the conventional form of backscatter coefficient given by:
# $$ \sigma o(dB) = \beta o(dB) + 10\log_10[\sin(ip)/\sin(ic)]
# where
#  - ip = is angle of incidencefor a particular pixel
#  - "ic" is angle of incidence for center of the image
#  - K = constant
#
# We have been given $ \sigma o $ directly in the data.

# Now coming to the features of $ \sigma o $
# Basically $\sigma o$ varieswith the surface on which the signal is scattered from.
# For example, for a particular angle of incidence, it varies like:
#
# - WATEr....SETTLEMENTS.....AGRICULTURE.....BARREN.....
#  - 1. HH: -27.001 ..... 2.70252 ..... -12.7952 ..... -17.257
#  - 2. HV: -28.035 ..... 20.2665 ..... -21.4471 ..... -20.019
#
# As you can see, the HH component varies a lot but HV doesn't.
# I don't have the data for scatter from ship, but being a metal object, it should vary differently as compared to ice object.

# WTF is HH HV?
# Ok, so this Sentinal Settalite is equivalent to RISTSAT (an Indian remote sensing Sat) and they only Transmit pings in H polarization,
# AND NOT IN V polarization. Those H-pings gets scattered, objects change their polarization and returns as a mix of H and V.
# Since Sentinel has only H transmitter, return signals are of the form of HH and HV only.
# Don't ask why VV is not given (because Sentinel don't have V-ping transmitter).

# Now coming to features, for the purpose of this demo code, I am extracting all two bands and taking avg of them as 3rd channel to create a 3-channel RGB equivalent.







#Generate the training data
#Create 3 bands having HH, HV and avg of both
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)



#Take a look at a iceberg
import plotly.offline as py
import plotly.graph_objs as go
# py.init_notebook_mode(connected=True)
def plotmy3d(c, name):

    data = [
        go.Surface(
            z=c
        )
    ]
    layout = go.Layout(
        title=name,
        autosize=False,
        width=700,
        height=700,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


plotmy3d(X_band_1[12,:,:], 'iceberg')

# That's a cool looking iceberg we have.
# Remember, in radar data, the shape of the iceberg is going to be like a mountain as shown in here.
# Since this is not a actual image but scatter from radar, the shape is going to have peaks and distortions like these.
# The shape of the ship is going to be like a point, may be like a elongated point.
# From here the structural differences arise and we can exploitthose differences using a CNN.
# It would be helpful if we can create composite images using the backscatter from radar.


plotmy3d(X_band_1[14,:,:], 'Ship')

# That's a ship, looks like a elongated point.
# We don't have much resolution in images to visualize the shape of the ship.
# However CNN is here to help.
# There are few papers on ship iceberg classification like this:
# http://elib.dlr.de/99079/2/2016_BENTES_Frost_Velotto_Tings_EUSAR_FP.pdf
# However, their data have much better resolution so I don't feel that the CNN they used would be suitable here.

# Get back to building a CNN using Keras.
# Much better frameworks than others.
# You will enjoy for sure.



# import tensorflow as tf

# https://shakeratos.tistory.com/m/9?category=715017
# pip install --upgrade tensorflow-gpu==1.4.0
# conda install -c anaconda cudnn
# conda install -c menpo opencv


# pip install --upgrade tensorflow==1.8.0
# pip install --upgrade keras==2.1.6
# pip install --upgrade theano==1.0.1
# pip install --upgrade numpy==1.13.1


# pip3 install --user tensorflow-gpu==2.3.0
# pip install --upgrade gast==0.2.2
# pip install --upgrade tensorboard==2.1.0
# pip install --upgrade keras==2.3.1

# conda install numpy pandas
# pip install sklearn
# pip install --upgrade keras
# conda install keras==2.3.1


#Import Keras.
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping






#define our model
def getModel():
    #Building the model
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    #Dense Layers
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Dense Layer 2
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))

    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)





target_train=train['is_iceberg']
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, target_train, random_state=1, train_size=0.75)





#Without denoising, core features.
import os
gmodel=getModel()
gmodel.fit(X_train_cv, y_train_cv,
          batch_size=24,
          epochs=50,
          verbose=1,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)




# Conclusion
# To increase the score, I have tried Speckie filtering, indicence angle normalization and other preprocessing and they don't seems to work.
# You may try and see but for me they are not giving any good results.

# You can't be on top-10 using this kernel, so here is one beautiful piece of information.
# The test dataset contain 8000 images. We can exploit this.
# We can do pseudo labelling to increase the predictions.
# Here is the article related to that:
# https://towardsdatascience.com/simple-explanation-of-semi-supervised-learning-and-pseudo-labeling-c2218e8c769b
# Upvote if you liked this kernel.


















from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
import numpy as np

image_dim = 784;
batch_size = 10;
input_img = Input(shape=(image_dim,))

encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(4, activation='relu')(encoded)

#decoding
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(image_dim, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

'''
saves the model weights after each epoch if the validation loss decreased
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


model_path = "weights.hdf5"
checkpointer = ModelCheckpoint(filepath=model_path, verbose=1)

autoencoder.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=100,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[checkpointer])

