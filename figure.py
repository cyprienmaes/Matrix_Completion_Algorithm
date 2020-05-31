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

    return matrix


data = pd.read_csv('ml1m_ratings.csv', sep='\t', encoding='latin-1',
                   usecols=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])

# +1 is the real size, as they are zero based
num_users = data['user_emb_id'].unique().max() + 1
num_items = data['movie_emb_id'].unique().max() + 1

# 10% of the full data are used for test.
# Kind of time intervals as netflix prize.
# The stratify is used to correctly split each user between train and test.
train_data, test_data = train_test_split(data,
                                         stratify=data['user_emb_id'],
                                         test_size=0.1,
                                         random_state=999613182)

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
"""
# RMSE of deep-autoencoder of three hidden layers function of epochs.
history = [None] * 7
history[0] = np.load('item_zero_128_128_128_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[1] = np.load('item_zero_256_128_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[2] = np.load('item_zero_256_256_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[3] = np.load('item_zero_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[4] = np.load('item_zero_512_256_512_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[5] = np.load('item_zero_512_512_512_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[6] = np.load('item_zero_512_1024_512_Adam_dropout_08.npy',allow_pickle='TRUE').item()

model = [None] * 7
model[0] = load_model('item_zero_128_128_128_Adam_dropout_08')
model[0].load_weights('item_zero_128_128_128_Adam_dropout_08.h5')
model[1] = load_model('item_zero_256_128_256_Adam_dropout_08')
model[1].load_weights('item_zero_256_128_256_Adam_dropout_08.h5')
model[2] = load_model('item_zero_256_256_256_Adam_dropout_08')
model[2].load_weights('item_zero_256_256_256_Adam_dropout_08.h5')
model[3] = load_model('item_zero_256_512_256_Adam_dropout_08')
model[3].load_weights('item_zero_256_512_256_Adam_dropout_08.h5')
model[4] = load_model('item_zero_512_256_512_Adam_dropout_08')
model[4].load_weights('item_zero_512_256_512_Adam_dropout_08.h5')
model[5] = load_model('item_zero_512_512_512_Adam_dropout_08')
model[5].load_weights('item_zero_512_512_512_Adam_dropout_08.h5')
model[6] = load_model('item_zero_512_1024_512_Adam_dropout_08')
model[6].load_weights('item_zero_512_1024_512_Adam_dropout_08.h5')

for hist in history:
    rmse = hist['masked_rmse_clip']
    val_rmse = hist['val_masked_rmse_clip']
    plt.plot(np.arange(20, len(rmse), 1), rmse[20:])
    # plt.plot(np.arange(20, len(val_rmse), 1), savgol_filter(val_rmse[20:], 21, 1))
    # plt.plot(np.arange(20, len(val_rmse), 1), val_rmse[20:])

plt.title('training rmse')

plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.grid()
plt.legend(['128x128x128', '256x128x256', '256x256x256',
            '256x512x256','512x256x512', '512x512x512',
            '512x1024x512'], loc='best')
plt.show()

for hist in history:
    loss = hist['loss']
    val_loss = hist['val_loss']
    # plt.plot(np.arange(200, len(loss), 1), loss[200:])
    plt.plot(np.arange(200, len(loss), 1), val_loss[200:])

plt.title('Loss of deep-autoencoder of three hidden layers for validation data function of epochs. \n'
          '[Adam optimizer (learning-rate=0.0001), dropout=0.8, regularizer=0.001, activation=SELU]')

plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['128x128x128', '256x128x256', '256x256x256',
            '256x512x256','512x256x512', '512x512x512',
            '512x1024x512'], loc='best')
plt.show()

result_loss = [None] * len(model)
result_rmse = [None] * len(model)
i=0
for mod in model:
    mod.compile(optimizer= tf.keras.optimizers.Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip])
    result = mod.evaluate(train_zero,test)
    result_loss[i] = result[0]
    result_rmse[i] = result[1]
    i = i + 1

plt.plot(np.arange(len(model)), result_rmse)
plt.title('RMSE  evaluation of deep-autoencoder of three hidden layers for test data. \n'
          '[Adam optimizer (learning-rate=0.0001), dropout=0.8, regularizer=0.001, activation=SELU]')
plt.ylabel('RMSE')
plt.grid()

plt.show()
"""
"""
# RMSE of deep-autoencoder 256x512x256 for different dropout function of epochs.
history = [None] * 6
history[0] = np.load('item_zero_256_512_256_Adam_dropout_00.npy',allow_pickle='TRUE').item()
history[1] = np.load('item_zero_256_512_256_Adam_dropout_02.npy',allow_pickle='TRUE').item()
history[2] = np.load('item_zero_256_512_256_Adam_dropout_04.npy',allow_pickle='TRUE').item()
history[3] = np.load('item_zero_256_512_256_Adam_dropout_06.npy',allow_pickle='TRUE').item()
history[4] = np.load('item_zero_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[5] = np.load('item_zero_256_512_256_Adam_dropout_09.npy',allow_pickle='TRUE').item()

model = [None] * 6
model[0] = load_model('item_zero_256_512_256_Adam_dropout_00')
model[0].load_weights('item_zero_256_512_256_Adam_dropout_00.h5')
model[1] = load_model('item_zero_256_512_256_Adam_dropout_02')
model[1].load_weights('item_zero_256_512_256_Adam_dropout_02.h5')
model[2] = load_model('item_zero_256_512_256_Adam_dropout_04')
model[2].load_weights('item_zero_256_512_256_Adam_dropout_04.h5')
model[3] = load_model('item_zero_256_512_256_Adam_dropout_06')
model[3].load_weights('item_zero_256_512_256_Adam_dropout_06.h5')
model[4] = load_model('item_zero_256_512_256_Adam_dropout_08')
model[4].load_weights('item_zero_256_512_256_Adam_dropout_08.h5')
model[5] = load_model('item_zero_256_512_256_Adam_dropout_09')
model[5].load_weights('item_zero_256_512_256_Adam_dropout_09.h5')

for hist in history:
    # rmse = hist['masked_rmse_clip']
    val_rmse = hist['val_masked_rmse_clip']
    # plt.plot(np.arange(20, len(rmse), 1), rmse[20:])
    # plt.plot(np.arange(20, len(val_rmse), 1), savgol_filter(val_rmse[20:], 21, 1))
    plt.plot(np.arange(20, len(val_rmse), 1), val_rmse[20:])

plt.title('rmse validation data')

plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.grid()
plt.legend(['dropout=0.0', 'dropout=0.2', 'dropout=0.4',
            'dropout=0.6','dropout=0.8', 'dropout=0.9'], loc='best')
plt.show()

for hist in history:
    loss = hist['loss']
    val_loss = hist['val_loss']
    plt.plot(np.arange(100, len(loss), 1), loss[100:])
    # plt.plot(np.arange(100, len(loss), 1), val_loss[100:])

plt.title('Loss of deep-autoencoder 256x512x256 for different droupout (training data) function of epochs. \n'
          '[Adam optimizer (learning-rate=0.0001), regularizer=0.001, activation=SELU]', fontsize=25)

plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['dropout=0.0', 'dropout=0.2', 'dropout=0.4',
            'dropout=0.6','dropout=0.8', 'dropout=0.9'], loc='best')
plt.show()

result_loss = [None] * len(model)
result_rmse = [None] * len(model)
dropout = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
i=0
for mod in model:
    mod.compile(optimizer= tf.keras.optimizers.Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip])
    result = mod.evaluate(train_zero,test)
    result_loss[i] = result[0]
    result_rmse[i] = result[1]
    i = i + 1

plt.plot(dropout, result_rmse)
plt.title('test rmse')

plt.ylabel('RMSE')
plt.xlabel('dropout')
plt.grid()

plt.show()
"""
"""
# RMSE of deep-autoencoder 256x128x256 for different re-feeding function of epochs.
history = [None] * 3
history[0] = np.load('item_zero_256_128_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[1] = np.load('item_zero_256_128_256_Adam_dropout_08_3.npy',allow_pickle='TRUE').item()
history[2] = np.load('item_zero_256_128_256_Adam_dropout_08_5.npy',allow_pickle='TRUE').item()

model = [None] * 3
model[0] = load_model('item_zero_256_128_256_Adam_dropout_08')
model[0].load_weights('item_zero_256_128_256_Adam_dropout_08.h5')
model[1] = load_model('item_zero_256_128_256_Adam_dropout_08_3')
model[1].load_weights('item_zero_256_128_256_Adam_dropout_08_3.h5')
model[2] = load_model('item_zero_256_128_256_Adam_dropout_08_5')
model[2].load_weights('item_zero_256_128_256_Adam_dropout_08_5.h5')

for hist in history:
    # rmse = hist['masked_rmse_clip']
    val_rmse = hist['val_masked_rmse_clip']
    # plt.plot(np.arange(0, len(rmse), 1), rmse[0:])
    # plt.plot(np.arange(20, len(val_rmse), 1), savgol_filter(val_rmse[20:], 21, 1))
    plt.plot(np.arange(0, len(val_rmse), 1), val_rmse[0:])

plt.title('RMSE of deep-autoencoder 256x128x256 for different re-feeding (validation data) function of epochs. \n'
          '[Adam optimizer (learning-rate=0.0001), dropout=0.8, regularizer=0.001, activation=SELU]')

plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.grid()
plt.legend(['re-feeding=0', 're-feeding=3', 're-feeding=5'], loc='best')
plt.show()

for hist in history:
    loss = hist['loss']
    val_loss = hist['val_loss']
    # plt.plot(np.arange(0, len(loss), 1), loss[0:])
    plt.plot(np.arange(100, len(loss), 1), val_loss[100:])

plt.title('Loss of deep-autoencoder 256x128x256 for different re-feeding (validation data) function of epochs. \n'
          '[Adam optimizer (learning-rate=0.0001), dropout=0.8, regularizer=0.001, activation=SELU]')

plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['re-feeding=0', 're-feeding=3', 're-feeding=5'], loc='best')
plt.show()

result_loss = [None] * len(model)
result_rmse = [None] * len(model)
re_feeding = [1, 3, 5]
i=0
for mod in model:
    mod.compile(optimizer= tf.keras.optimizers.Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip])
    result = mod.evaluate(train_zero,test)
    result_loss[i] = result[0]
    result_rmse[i] = result[1]
    i = i + 1

plt.plot(re_feeding, result_rmse)
plt.title('RMSE  evaluation of deep-autoencoder 256x128x256 for test data for different re-feeding. \n'
          '[Adam optimizer (learning-rate=0.0001), dropout=0.8, regularizer=0.001, activation=SELU]')

plt.ylabel('RMSE')
plt.xlabel('re-feeding')
plt.grid()

plt.show()
"""
"""
# RMSE of deep-autoencoder with different number of layers of size 256 function of epochs.
history = [None] * 4
history[0] = np.load('item_zero_256_256_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[1] = np.load('item_zero_256_256_256_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[2] = np.load('item_zero_256_256_256_256_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[3] = np.load('item_zero_256_256_256_256_256_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()

model = [None] * 4
model[0] = load_model('item_zero_256_256_256_Adam_dropout_08')
model[0].load_weights('item_zero_256_256_256_Adam_dropout_08.h5')
model[1] = load_model('item_zero_256_256_256_256_Adam_dropout_08')
model[1].load_weights('item_zero_256_256_256_256_Adam_dropout_08.h5')
model[2] = load_model('item_zero_256_256_256_256_256_Adam_dropout_08')
model[2].load_weights('item_zero_256_256_256_256_256_Adam_dropout_08.h5')
model[3] = load_model('item_zero_256_256_256_256_256_256_Adam_dropout_08')
model[3].load_weights('item_zero_256_256_256_256_256_256_Adam_dropout_08.h5')

for hist in history:
    rmse = hist['masked_rmse_clip']
    val_rmse = hist['val_masked_rmse_clip']
    # plt.plot(np.arange(50, len(rmse), 1), rmse[50:])
    # plt.plot(np.arange(20, len(val_rmse), 1), savgol_filter(val_rmse[20:], 21, 1))
    plt.plot(np.arange(50, len(val_rmse), 1), val_rmse[50:])

plt.title('validation rmse')

plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.grid()
plt.legend(['256x256x256', '256x256x256x256', '256x256x256x256x256',
            '256x256x256x256x256x256'], loc='best')
plt.show()

for hist in history:
    loss = hist['loss']
    val_loss = hist['val_loss']
    plt.plot(np.arange(100, len(loss), 1), loss[100:])
    # plt.plot(np.arange(200, len(loss), 1), val_loss[200:])

plt.title('Loss of deep-autoencoder with different number of layers of size 256 (training data) function of epochs. \n'
          '[Adam optimizer (learning-rate=0.0001), dropout=0.8, regularizer=0.001, activation=SELU]')

plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['256x256x256', '256x256x256x256', '256x256x256x256x256',
            '256x256x256x256x256x256'], loc='best')
plt.show()

result_loss = [None] * len(model)
result_rmse = [None] * len(model)
re_feeding = [3, 4, 5, 6]
i=0
for mod in model:
    mod.compile(optimizer= tf.keras.optimizers.Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip])
    result = mod.evaluate(train_zero,test)
    result_loss[i] = result[0]
    result_rmse[i] = result[1]
    i = i + 1

plt.plot(re_feeding, result_rmse)
plt.title('test rmse')

plt.ylabel('RMSE')
plt.xlabel('number of layers')
plt.grid()

plt.show()
"""
"""
# RMSE of deep-autoencoder 256x512x256 with different optimizers function of epochs.
history = [None] * 5
history[0] = np.load('item_zero_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[1] = np.load('item_zero_256_512_256_Rmsprop_dropout_08.npy',allow_pickle='TRUE').item()
history[2] = np.load('item_zero_256_512_256_Adadelta_dropout_08.npy',allow_pickle='TRUE').item()
history[3] = np.load('item_zero_256_512_256_SGD_dropout_08.npy',allow_pickle='TRUE').item()
history[4] = np.load('item_zero_256_512_256_Adagrad_dropout_08.npy',allow_pickle='TRUE').item()

model = [None] * 5
model[0] = load_model('item_zero_256_512_256_Adam_dropout_08')
model[0].load_weights('item_zero_256_512_256_Adam_dropout_08.h5')
model[1] = load_model('item_zero_256_512_256_Rmsprop_dropout_08')
model[1].load_weights('item_zero_256_512_256_Rmsprop_dropout_08.h5')
model[2] = load_model('item_zero_256_512_256_Adadelta_dropout_08')
model[2].load_weights('item_zero_256_512_256_Adadelta_dropout_08.h5')
model[3] = load_model('item_zero_256_512_256_SGD_dropout_08')
model[3].load_weights('item_zero_256_512_256_SGD_dropout_08.h5')
model[4] = load_model('item_zero_256_512_256_Adagrad_dropout_08')
model[4].load_weights('item_zero_256_512_256_Adagrad_dropout_08.h5')

for hist in history:
    rmse = hist['masked_rmse_clip']
    val_rmse = hist['val_masked_rmse_clip']
    plt.plot(np.arange(50, len(rmse), 1), rmse[50:])
    # plt.plot(np.arange(20, len(val_rmse), 1), savgol_filter(val_rmse[20:], 21, 1))
    # plt.plot(np.arange(50, len(val_rmse), 1), val_rmse[50:])

plt.title('RMSE of deep-autoencoder 256x512x256 with different optimizers (training data) function of epochs. \n'
          'learning-rate=0.0001, dropout=0.8, regularizer=0.001, activation=SELU]')

plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.grid()
plt.legend(['Adam', 'RmsProp', 'Adadelta', 'SGD',
            'Adagrad'], loc='best')
plt.show()

for hist in history:
    loss = hist['loss']
    val_loss = hist['val_loss']
    plt.plot(np.arange(100, len(loss), 1), loss[100:])
    # plt.plot(np.arange(200, len(loss), 1), val_loss[200:])

plt.title('Loss of deep-autoencoder 256x512x256 with different optimizers (training data) function of epochs. \n'
          '[learning-rate=0.0001, dropout=0.8, regularizer=0.001, activation=SELU]')

plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['Adam', 'RmsProp', 'Adadelta', 'SGD',
            'Adagrad'], loc='best')
plt.show()
"""

# RMSE of deep-autoencoder 256x512x256 with different learning rate function of epochs.
history = [None] * 3
history[0] = np.load('item_zero_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[1] = np.load('item_zero_256_512_256_Adam_dropout_08_001.npy',allow_pickle='TRUE').item()
history[2] = np.load('item_zero_256_512_256_Adam_dropout_08_01.npy',allow_pickle='TRUE').item()

model = [None] * 3
model[0] = load_model('item_zero_256_512_256_Adam_dropout_08')
model[0].load_weights('item_zero_256_512_256_Adam_dropout_08.h5')
model[1] = load_model('item_zero_256_512_256_Adam_dropout_08_001')
model[1].load_weights('item_zero_256_512_256_Adam_dropout_08_001.h5')
model[2] = load_model('item_zero_256_512_256_Adam_dropout_08_01')
model[2].load_weights('item_zero_256_512_256_Adam_dropout_08_01.h5')

for hist in history:
    rmse = hist['masked_rmse_clip']
    val_rmse = hist['val_masked_rmse_clip']
    # plt.plot(np.arange(0, len(rmse), 1), rmse[0:])
    # plt.plot(np.arange(20, len(val_rmse), 1), savgol_filter(val_rmse[20:], 21, 1))
    plt.plot(np.arange(0, len(val_rmse), 1), val_rmse[0:])

plt.title('validation rmse')

plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.grid()
plt.legend(['learning_rate=0.0001', 'learning_rate=0.001', 'learning_rate=0.01'], loc='best')
plt.show()

for hist in history:
    loss = hist['loss']
    val_loss = hist['val_loss']
    # plt.plot(np.arange(0, len(loss), 1), loss[0:])
    plt.plot(np.arange(0, len(loss), 1), val_loss[0:])

plt.title('validation rmse')

plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['learning_rate=0.0001', 'learning_rate=0.001', 'learning_rate=0.01'], loc='best')
plt.show()

result_loss = [None] * len(model)
result_rmse = [None] * len(model)
learning_rate = [0.0001, 0.001, 0.01]
i=0
for mod in model:
    mod.compile(optimizer= tf.keras.optimizers.Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip])
    result = mod.evaluate(train_zero,test)
    result_loss[i] = result[0]
    result_rmse[i] = result[1]
    i = i + 1

plt.semilogx(learning_rate, result_rmse)
plt.title('test rmse')

plt.ylabel('RMSE')
plt.xlabel('Learning rate')
plt.grid()

plt.show()

"""
# RMSE of deep-autoencoder 256x512x256 with different constant rate function of epochs.
history = [None] * 7
history[0] = np.load('item_zero_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[1] = np.load('item_one_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[2] = np.load('item_two_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[3] = np.load('item_three_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[4] = np.load('item_four_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[5] = np.load('item_five_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()
history[6] = np.load('item_average_256_512_256_Adam_dropout_08.npy',allow_pickle='TRUE').item()

model = [None] * 7
model[0] = load_model('item_zero_256_512_256_Adam_dropout_08')
model[0].load_weights('item_zero_256_512_256_Adam_dropout_08.h5')
model[1] = load_model('item_one_256_512_256_Adam_dropout_08')
model[1].load_weights('item_one_256_512_256_Adam_dropout_08.h5')
model[2] = load_model('item_two_256_512_256_Adam_dropout_08')
model[2].load_weights('item_two_256_512_256_Adam_dropout_08.h5')
model[3] = load_model('item_three_256_512_256_Adam_dropout_08')
model[3].load_weights('item_three_256_512_256_Adam_dropout_08.h5')
model[4] = load_model('item_four_256_512_256_Adam_dropout_08')
model[4].load_weights('item_four_256_512_256_Adam_dropout_08.h5')
model[5] = load_model('item_five_256_512_256_Adam_dropout_08')
model[5].load_weights('item_five_256_512_256_Adam_dropout_08.h5')
model[6] = load_model('item_average_256_512_256_Adam_dropout_08')
model[6].load_weights('item_average_256_512_256_Adam_dropout_08.h5')

for hist in history:
    rmse = hist['masked_rmse_clip']
    val_rmse = hist['val_masked_rmse_clip']
    # plt.plot(np.arange(0, len(rmse), 1), rmse[0:])
    # plt.plot(np.arange(20, len(val_rmse), 1), savgol_filter(val_rmse[20:], 21, 1))
    plt.plot(np.arange(100, len(val_rmse), 1), val_rmse[100:])

plt.title('RMSE of deep-autoencoder 256x512x256 with different constant rate (validation data) function of epochs. \n'
          '[Adam optimizer (learning-rate=0.0001), dropout=0.8, regularizer=0.001, activation=SELU]')

plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.grid()
plt.legend(['zero', 'one', 'two', 'three', 'four', 'five', 'average'], loc='best', fontsize='x-large')
plt.show()

for hist in history:
    loss = hist['loss']
    val_loss = hist['val_loss']
    # plt.plot(np.arange(0, len(loss), 1), loss[0:])
    plt.plot(np.arange(100, len(loss), 1), val_loss[100:])

plt.title('Loss of deep-autoencoder 256x512x256 with different constant rate (validation data) function of epochs. \n'
          '[Adam optimizer (learning-rate=0.0001), dropout=0.8, regularizer=0.001, activation=SELU]')

plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['zero', 'one', 'two', 'three', 'four', 'five', 'average'], loc='best',  fontsize='x-large')
plt.show()
"""