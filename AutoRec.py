import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.layers import Input, Dense, Embedding, Flatten, Dropout, merge, Activation, BatchNormalization, LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from sklearn import preprocessing
# from tensorflow.python.keras.utils import plot_model

df = pd.read_csv('ml1m_ratings.csv',
                 sep='\t',
                 encoding='latin-1',
                 usecols=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])

#+1 is the real size, as they are zero based
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


def dataPreprocessor(rating_df, num_users, num_items, init_value=0, average=False):
    """
        INPUT:
            data: pandas DataFrame. columns=['userID', 'itemID', 'rating' ...]
            num_row: int. number of users
            num_col: int. number of items

        OUTPUT:
            matrix: 2D numpy array.
    """
    if average:
        matrix = np.full((num_users, num_items), 0.0)
        for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
            matrix[userID, itemID] = rating

        avergae = np.true_divide(matrix.sum(1), np.maximum((matrix != 0).sum(1), 1))
        inds = np.where(matrix == 0)
        matrix[inds] = np.take(avergae, inds[0])

    else:
        matrix = np.full((num_users, num_items), init_value)
        for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
            matrix[userID, itemID] = rating
    return matrix

# Creating a sparse pivot table with users in rows and items in columns
users_items_matrix_train_zero = dataPreprocessor(train_df, num_users, num_movies, 0)
users_items_matrix_train_one = dataPreprocessor(train_df, num_users, num_movies, 1)
users_items_matrix_train_two = dataPreprocessor(train_df, num_users, num_movies, 2)
users_items_matrix_train_four = dataPreprocessor(train_df, num_users, num_movies, 4)
users_items_matrix_train_three = dataPreprocessor(train_df, num_users, num_movies, 3)
users_items_matrix_train_five = dataPreprocessor(train_df, num_users, num_movies, 5)
users_items_matrix_validate = dataPreprocessor(validate_df, num_users, num_movies, 0)
users_items_matrix_test = dataPreprocessor(test_df, num_users, num_movies, 0)

users_items_matrix_train_average = dataPreprocessor(train_df, num_users, num_movies, average=True)

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
    y_pred = K.clip(y_pred, 1, 5)
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_mse

# Test custom cost function
y_pred = K.constant([[ 1, 1, 1, 1],
                     [ 1, 1, 1, 10],
                     [ 1, 1, 1, 3],
                     [ 1, 1, 1, 3],
                     [ 1, 1, 1, 3],
                     [ 1, 1, 1, 3]])
y_pred = K.constant([[ 1, 1, 1, 3]])
y_true = K.constant([[ 1, 1, 1, 1],
                     [ 1, 1, 1, 1],
                     [ 0, 1, 1, 1],
                     [ 0, 0, 1, 1],
                     [ 0, 0, 0, 1],
                     [ 0, 0, 0, 0]])
y_true = K.constant([[ 0, 0, 1, 1]])
true = K.eval(y_true)
pred = K.eval(y_pred)
loss = K.eval(masked_se(y_true, y_pred))
rmse = K.eval(masked_rmse(y_true, y_pred))

for i in range(true.shape[0]):
    print(true[i], pred[i], loss[i], rmse[i], sep='\t')


def AutoRec(X, reg, first_activation, last_activation):
    '''
    AutoRec
        INPUT:
          X: #_user X #_item matrix
          reg: L2 regularization parameter
          first_activation: activation function for first dense layer
          last_activation: activation function for second dense layer

        OUTPUT:
          Keras model

    '''
    input_layer = x = Input(shape=(X.shape[1],),
                            name='UserRating')
    x = Dense(500, activation=first_activation,
              name='LatentSpace',
              kernel_regularizer=regularizers.l2(reg))(x)
    output_layer = Dense(X.shape[1],
                         activation=last_activation,
                         name='UserScorePred',
                         kernel_regularizer=regularizers.l2(reg))(x)
    model = Model(input_layer, output_layer)

    return model


# Build model

AutoRec = AutoRec(users_items_matrix_train_zero, 0.0005, 'elu', 'elu')

AutoRec.compile(optimizer=Adam(lr=0.0001), loss=masked_mse, metrics=[masked_rmse_clip])

AutoRec.summary()

hist_Autorec = AutoRec.fit(x=users_items_matrix_train_average, y=users_items_matrix_train_zero,
                  epochs=500,
                  batch_size=256,
                  verbose = 2,
                  validation_data=[users_items_matrix_train_average, users_items_matrix_validate])

# plot_model(AutoRec, to_file='AutoRec.png')

show_rmse(hist_Autorec, 30)

show_error(hist_Autorec, 50)

test_result = AutoRec.evaluate(users_items_matrix_train_average, users_items_matrix_test)

predict_autorec = AutoRec.predict(users_items_matrix_train_zero)

