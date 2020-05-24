import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.layers import Input, Dense, Embedding, Flatten, Dropout, merge, Activation
from tensorflow.python.keras.layers import BatchNormalization, LeakyReLU, add, concatenate
from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import initializers
import warnings

warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.sparse import csr_matrix
import tensorflow as tf
from sklearn import preprocessing
from keras.utils import plot_model

# Data Preprocessing
df = pd.read_csv('ml100k_ratings.csv',
                 sep='\t',
                 encoding='latin-1',
                 usecols=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])
# Normalize movie lens data base
df['rating'] = df['rating']/5.0
# +1 is the real size, as they are zero based
num_users = df['user_emb_id'].unique().max() + 1
num_movies = df['movie_emb_id'].unique().max() + 1
df.head(5)

train_df, test_df = train_test_split(df,
                                     stratify=df['user_emb_id'],
                                     test_size=0.1,
                                     random_state=999613182)

train_df.head(5)

train_df, validate_df = train_test_split(train_df,
                                         stratify=train_df['user_emb_id'],
                                         test_size=0.1,
                                         random_state=999613182)


def data_pre_processor(rating_df, num_row, num_col, init_value=0, average=False):
    """Construction of a matrix with the users as rows and items as columns.

    The matrix can be chosen with different constant for the empty part. The matrix can be also
    constructed with the average for each existing items.

    Parameters
    ----------
    :param rating_df: pandas DataFrame.
        colums=['userID', 'itemID', 'rating', ...]
    :param num_row: int.
        Number of users
    :param num_col: int.
        Number of items
    :param init_value: int.
        Value of empty entry
    :param average: bool.
        Average the sum of matrix columns by the number of true element
    Return
    ------
    :return: 2D numpy array.
    """
    if average:
        matrix = np.full((num_row, num_col), 0.0)
        for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
            matrix[userID, itemID] = rating

        avg = np.true_divide(matrix.sum(1), np.maximum((matrix != 0).sum(1), 1))
        indx = np.where(matrix == 0)
        matrix[indx] = np.take(avg, indx[0])

    else:
        matrix = np.full((num_row, num_col), init_value)
        for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
            matrix[userID, itemID] = rating

    return matrix


# Creating a sparse pivot table with users in rows and items in columns
users_items_matrix_train_zero = data_pre_processor(train_df, num_users, num_movies, 0.0)
# users_items_matrix_train_one = data_pre_processor(train_df, num_users, num_movies, 1)
# users_items_matrix_train_two = data_pre_processor(train_df, num_users, num_movies, 2)
# users_items_matrix_train_three = data_pre_processor(train_df, num_users, num_movies, 3)
# users_items_matrix_train_four = data_pre_processor(train_df, num_users, num_movies, 4)
# users_items_matrix_train_five = data_pre_processor(train_df, num_users, num_movies, 5)
users_items_matrix_validate = data_pre_processor(validate_df, num_users, num_movies, 0.0)
users_items_matrix_test = data_pre_processor(test_df, num_users, num_movies, 0.0)

users_items_matrix_average = data_pre_processor(train_df, num_users, num_movies, average=True)


# Utility Function
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


def load_model(name):
    """Load json and create model."""
    model_file = open('{}.json'.format(name), 'r')
    loaded_model_json = model_file.read()
    model_file.close()
    # keras export model from json file
    loaded_model = model_from_json(loaded_model_json)
    # Load weights into new model
    loaded_model.load_weights("{}.h5".format(name))
    print("Loaded model from disk")
    return loaded_model


def save_model(name, model):
    """Serialize model to JSON."""
    model_json = model.to_json()
    with open("{}.json".format(name), "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("{}.h5".format(name))
    print("Saved model to disk")


def masked_se(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1)
    return masked_mse


def masked_mse(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    # Normalized error
    masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
    return masked_mse


def masked_rmse(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    # Normalized error
    masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_mse


def masked_rmse_clip(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_pred = K.clip(y_pred, 0.0, 1.0)
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    # Normalized error
    masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_mse


# Test custom cost function
# y_prediction = K.constant([[0.4, 0.8, 1.0, 1.0],
#                            [0.8, 0.8, 0.2, 0.2],
#                            [1.0, 1.0, 1.0, 0.6],
#                            [0.0, 0.0, 1.0, 0.4],
#                            [0.0, 0.0, 0.0, 0.2],
#                            [0.0, 0.4, 0.8, 0.6]])
y_prediction = K.constant([[1.0, 1.0, 1.0, 0.4]])
# y_true = K.constant([[0.4, 0.8, 0.4, 0.4],
#                      [0.8, 0.8, 0.2, 0.2],
#                      [0.0, 1.0, 1.0, 0.6],
#                      [0.0, 0.0, 1.0, 0.4],
#                      [0.0, 0.0, 0.0, 0.2],
#                      [0.0, 0.0, 0.0, 0.0]])
y_true = K.constant([[0.0, 0.6, 0.2, 0.4]])
true = K.eval(y_true)
pred = K.eval(y_prediction)
loss = K.eval(masked_se(y_true, y_prediction))
rmse = K.eval(masked_rmse_clip(y_true, y_prediction))

for i in range(true.shape[0]):
    print(true[i], pred[i], loss[i], rmse[i], sep='\t')

# Particular weight initialization
def custom_random_initializer(shape, dtype=None):
    if len(shape) == 1:
        rand = np.random.rand(shape[0]) - 0.5
        s = np.sqrt(6.0 / (shape[0] - 1))
    else:
        rand = np.random.rand(shape[0], shape[1]) - 0.5
        s = np.sqrt(6.0 / (shape[0] + shape[1] - 1))
    rand = 2.0 * 4.0 * rand
    rand = rand * s
    w = tf.convert_to_tensor(rand, dtype=tf.float32)
    return w

def tanh_opt(x):
    return 1.7159*K.tanh(2.0/3.0*x)

get_custom_objects().update({'tanh_opt': Activation(tanh_opt)})

# AutoRec
def auto_rec(matrix, latent_dim, reg, first_activation, last_activation):
    input_layer = x = Input(shape=(matrix.shape[1],),
                        name='UserRating')
    x = Dense(latent_dim,
              activation=first_activation,
              kernel_initializer=custom_random_initializer,
              bias_initializer=custom_random_initializer,
              name='LatentSpace',
              kernel_regularizer=regularizers.l2(reg))(x)
    output_layer = Dense(matrix.shape[1],
                         activation=last_activation,
                         kernel_initializer=custom_random_initializer,
                         bias_initializer=custom_random_initializer,
                         name='UserScorePred',
                         kernel_regularizer=regularizers.l2(reg))(x)
    model = Model(input_layer, output_layer)
    return model


def auto_rec_lrelu(matrix, reg):
    input_layer = x = Input(shape=(matrix.shape[1],),
                        name='UserRating')
    x = Dense(500,
              name='LatentSpace',
              kernel_regularizer=regularizers.l2(reg))(x)
    x = LeakyReLU()(x)
    output_layer = Dense(matrix.shape[1],
                         activation='linear',
                         name='UserScorePred',
                         kernel_regularizer=regularizers.l2(reg))(x)
    model = Model(input_layer, output_layer)
    return model


# Build model
tf.compat.v1.disable_eager_execution()
autorec = auto_rec(users_items_matrix_train_zero, 50, 0.001,'linear','sigmoid')
autorec.compile(optimizer = RMSprop(lr=0.0001), loss=masked_rmse, metrics=[masked_rmse_clip])
autorec.summary()

hist_autorec = autorec.fit(x=users_items_matrix_average, y=users_items_matrix_train_zero,
                           epochs=700,
                           batch_size=256,
                           verbose=2,
                           validation_data=[users_items_matrix_average, users_items_matrix_validate])

tf.keras.utils.plot_model(autorec, to_file='AutoRec.png')

show_rmse(hist_autorec,30)

show_error(hist_autorec,50)

test_result = autorec.evaluate(users_items_matrix_average, users_items_matrix_test)

predict_autorec = autorec.predict(users_items_matrix_train_zero)

print("Application finished")
