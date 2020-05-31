import pandas as pd
import numpy as np
import tensorflow  as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

history=np.load('item_zero_256_128_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()

def create_matrix_data(data, num_users, num_items, init_value=0, avg=False):
    """ Create a matrix data with ratings knowing the number of users and items.

        The matrix is created thanks to the id of each users and each items.

        PARAMETERS:
        -----------
        data: pandas DataFrame.
            columns=['userID', 'itemID', 'rating' ...]
        num_users: int.
            number of users (row matrix)
        num_items: int.
            number of items (column matrix)
        init_value: float.
            constant that are place into the missing entries
        avg: bool.
            the constant is replaced by the average of the notation for
            each users.

        RETURN:
        -------
        matrix: 2D numpy array.
            matrix R(i,j) used into the autoencoder neural network.

    """
    if avg:
        matrix = np.full((num_users, num_items), 0.0)
        for (_, userID, itemID, rating, timestamp) in data.itertuples():
            matrix[userID, itemID] = rating
        average = np.true_divide(matrix.sum(1), np.maximum((matrix != 0).sum(1), 1))
        inds = np.where(matrix == 0)
        matrix[inds] = np.take(average, inds[0])

    else:
        matrix = np.full((num_users, num_items), float(init_value))
        for (_, userID, itemID, rating, timestamp) in data.itertuples():
            matrix[userID, itemID] = rating

    print("First row and 20 first columns :")
    print(matrix[0, 0:20])
    return matrix


data = pd.read_csv('ml1m_ratings.csv', sep='\t', encoding='latin-1',
                   usecols=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])

# +1 is the real size, as they are zero based
num_users = data['user_emb_id'].unique().max() + 1
num_items = data['movie_emb_id'].unique().max() + 1

print('Data:')
print(data.head(5))

# 10% of the full data are used for test.
# Kind of time intervals as netflix prize.
# The stratify is used to correctly split each user between train and test.
train_data, test_data = train_test_split(data,
                                         stratify=data['user_emb_id'],
                                         test_size=0.1,
                                         random_state=999613182)

print('Train data:')
print(train_data.head(5))

# 10% of the train data are used for validation and tuned the parameters if
# necessary.
train_data, validate_data = train_test_split(train_data,
                                             stratify=train_data['user_emb_id'],
                                             test_size=0.1,
                                             random_state=999613182)

# Creating sparse matrix with different constants for missing entries.
train_zero = create_matrix_data(train_data, num_users, num_items, 0).T
train_one = create_matrix_data(train_data, num_users, num_items, 1).T
train_two = create_matrix_data(train_data, num_users, num_items, 2).T
train_four = create_matrix_data(train_data, num_users, num_items, 4).T
train_three = create_matrix_data(train_data, num_users, num_items, 3).T
train_five = create_matrix_data(train_data, num_users, num_items, 5).T
train_average = create_matrix_data(train_data, num_users, num_items, avg=True).T

validate = create_matrix_data(validate_data, num_users, num_items, 0).T
test = create_matrix_data(test_data, num_users, num_items, 0).T


def deep_autoencoders_model(matrix, layers, activation, last_activation,
                            re_feeding, dropout, regularizer_encode,
                            regularizer_decode):
    """ Construction of the deep autoencoders for collaborative filtering.

        The deep autoencoders is constructed with classical fully connected layers.
        The number of hidden layers can be chosen by the user as the different
        activation functions between each layer. A dropout rate can be applied
        after the latent layer to avoid overfitting. Moreover, re-feefing algorithm
        is also possible with a number of updates. The parameters are initialized
        with the Xavier initializer (or glorot uniform) which is the default one
        in keras.

        PARAMETERS:
        -----------
        matrix: 2D numpy array.
            Sparse matrix to complete.
        layers: List.
            each element is the number of neuron for a layer
        activation: List.
            each element is the activation function to use between layers except for
            the last.
        last_activation: str.
            activation function for the last dense layer
        re_feeding: int.
            number of re_feeding updates
        dropout: float.
            dropout rate between 0 and 1
        regularizer_encode: float.
            regularizer for encoder
        regularizer_decode: float.
            regularizer for decoder

        RETURN:
        -------
        model: keras Model.
            configuration of the model to use
    """
    x = [None]*(len(layers)+2)
    # Input
    input_layer = new_dense = Input(shape=(matrix.shape[1],), name='sparse_ratings')
    num_enc = int(len(layers)/2)
  
    # Encoder
    for i in range(num_enc):
      x[i] = Dense(layers[i], 
                   activation=activation[i],
                   name='encoded_layer{}'.format(i), 
                   kernel_regularizer=l2(regularizer_encode))
  
  
    # Latent layer
    x[num_enc] = Dense(layers[num_enc], 
                       activation=activation[num_enc], 
                       name='latent_layer', 
                       kernel_regularizer=l2(regularizer_encode))
  
    # Dropout rate
    x[num_enc+1] = Dropout(rate = dropout)
  
    # Decoder
    for i in range(num_enc+1,len(layers)):
      x[i+1] = Dense(layers[i], 
                activation=activation[i], 
                name='decoded_layerr{}'.format(i), 
                kernel_regularizer=l2(regularizer_decode))
  
    # Output
    output_layer = x[len(layers)+1] = Dense(matrix.shape[1],
                              activation=last_activation, 
                              name='predict_ratings', 
                              kernel_regularizer=l2(regularizer_decode))
  
    # Re-feeding algorithm
  
    for j in range(re_feeding):
      for layer in x:
        new_dense = layer(new_dense)

    model = Model(input_layer, new_dense)

    return model


def show_error(history, skip):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(np.arange(skip, len(loss), 1), loss[skip:])
    plt.plot(np.arange(skip, len(loss), 1), val_loss[skip:])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


def show_rmse(history, skip):
    rmse = history.history['masked_rmse_clip']
    val_rmse = history.history['val_masked_rmse_clip']
    plt.plot(np.arange(skip, len(rmse), 1), rmse[skip:])
    plt.plot(np.arange(skip, len(val_rmse), 1), val_rmse[skip:])
    plt.title('model train vs validation masked_rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


def masked_mse(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
    return masked_mse


def masked_rmse_clip(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_pred = K.clip(y_pred, 1, 5)
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_mse

def load_model(name):
  # load json and create model
  model_file = open('{}.json'.format(name), 'r')
  loaded_model_json = model_file.read()
  model_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("{}.h5".format(name))
  print("Loaded model from disk")
  return loaded_model

def save_model(name, model):
  # # serialize model to JSON
  model_json = model.to_json()
  with open("{}.json".format(name), "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("{}.h5".format(name))
  print("Saved model to disk")

def save_hist_model(name, hist_model):
  # # serialize model to JSON
  hist_df = pd.DataFrame(hist_model.history)
  # save to json:
  hist_json_file = "{}.json".format(name)
  with open(hist_json_file, mode='w') as f:
      hist_df.to_json(f)
  print("Saved historic model to disk")

def load_hist_model(name):
  # load json and create model
  model_file = open('{}.json'.format(name), 'r')
  loaded_model_json = model_file.read()
  model_file.close()
  print("Loaded model from disk")
  return loaded_model_json

tf.compat.v1.disable_eager_execution()

layers = [256, 512, 256]
# layers = [512, 256, 128, 256, 512]
# layers = [512, 256, 512]
# layers = [128, 256, 512, 256, 128]
# layers = [512, 512, 512]
dropout = 0.0
re_feeding = 1
# activation = 'sigmoid'
# last_activation = 'linear'
activation = ["linear", "linear", "selu"]
last_activation = 'selu'
regularizer_encode = 0.0005
regularizer_decode = 0.0005

deep_ae_cf = deep_autoencoders_model(train_zero,
                                     layers,
                                     activation,
                                     last_activation,
                                     re_feeding,
                                     dropout,
                                     regularizer_encode,
                                     regularizer_decode)

deep_ae_cf.compile(optimizer= Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip])
deep_ae_cf.summary()

hist_deep_ae_cf = deep_ae_cf.fit(x=train_zero, y=train_zero,
                                 epochs=500,
                                 batch_size=256,
                                 validation_data=[train_zero, validate], verbose=2)

show_error(hist_deep_ae_cf, 100)
show_rmse(hist_deep_ae_cf, 100)

test_result = deep_ae_cf.evaluate(train_zero,test)

predict_deep = deep_ae_cf.predict(train_zero)
print(predict_deep[40,0:20])
