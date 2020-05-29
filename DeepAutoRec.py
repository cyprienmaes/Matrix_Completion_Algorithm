import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense, Embedding, Flatten, Dropout, merge, Activation
from keras.models import Model, model_from_json
from keras import backend as K
from keras import regularizers
import warnings

warnings.filterwarnings('ignore')

# Data Preprocessing
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

def save_model(name, model):
    # # serialize model to JSON
    model_json = model.to_json()
    with open("{}.json".format(name), "w") as json_file:
      json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{}.h5".format(name))
    print("Saved model to disk")

def load_history(name):
    # load json and create model
    hist_file = open('{}.json'.format(name), 'r')
    loaded_hist_json = hist_file.read()
    hist_file.close()
    loaded_hist = pd.read_json(loaded_hist_json)
    # load weights into new model
    # loaded_model.load_weights("{}.h5".format(name))
    print("Loaded history from disk")
    return loaded_hist

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
arr1 = np.array([[ 1, 1, 1, 1],
                     [ 1, 1, 1, 10],
                     [ 1, 1, 1, 3],
                     [ 1, 1, 1, 3],
                     [ 1, 1, 1, 3],
                     [ 1, 1, 1, 3]])
y_pred = K.constant(arr1)
arr2 = np.array([[ 1, 1, 1, 3]])
y_pred = K.constant(arr2)

arr3 = np.array([[ 1, 1, 1, 1],
                     [ 1, 1, 1, 1],
                     [ 0, 1, 1, 1],
                     [ 0, 0, 1, 1],
                     [ 0, 0, 0, 1],
                     [ 0, 0, 0, 0]])
y_true = K.constant(arr3)
arr4 = np.array([[ 0, 0, 1, 1]])
y_true = K.constant(arr4)

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
    x = Dense(512, activation=first_activation,
              name='LatentSpace',
              kernel_regularizer=regularizers.l2(reg))(x)
    output_layer = Dense(X.shape[1],
                         activation=last_activation,
                         name='UserScorePred',
                         kernel_regularizer=regularizers.l2(reg))(x)
    model = Model(input_layer, output_layer)

    return model


# Build model

activation = ['linear', 'elu', 'selu']
lamb = [0.0005, 0.005]

for a in activation:
    print(a)
    for la in lamb:
        print(la)
        AutoRecM = AutoRec(users_items_matrix_train_zero.T, la, a, 'linear')

        AutoRecM.compile(optimizer=RMSprop(lr=0.001), loss=masked_mse, metrics=[masked_rmse_clip])

        AutoRecM.summary()

        hist_Autorec = AutoRecM.fit(x=users_items_matrix_train_three.T, y=users_items_matrix_train_zero.T,
                                   epochs=500,
                                   batch_size=256,
                                   verbose=2,
                                   validation_data=[users_items_matrix_train_three.T, users_items_matrix_validate.T])

        hist_df = pd.DataFrame(hist_Autorec.history)

        st = str(la * 2)

        # save to json:
        hist_json_file = 'history-1m-512-'+st[2:]+'-'+a[0]+'l-RMSprop-001.json'
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)

        # or save to csv:
        hist_csv_file = 'history-1m-512-'+st[2:]+'-'+a[0]+'l-RMSprop-001.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        K.clear_session()


        print('Finished !')

# AutoRec = AutoRec(users_items_matrix_train_zero.T, 0.05, 'selu', 'linear')
#
# AutoRec.compile(optimizer=Adam(lr=0.001), loss=masked_mse, metrics=[masked_rmse_clip])
#
# AutoRec.summary()
#
# hist_Autorec = AutoRec.fit(x=users_items_matrix_train_three.T, y=users_items_matrix_train_zero.T,
#                   epochs=500,
#                   batch_size=512,
#                   verbose = 2,
#                   validation_data=[users_items_matrix_train_three.T, users_items_matrix_validate.T])

# plot_model(AutoRec, to_file='AutoRec.png')

# Use 3 as initial value
# Compare 500  and 30

# with open('/trainHistoryDict', 'wb') as file_pi:
#     np.pickle.dump(hist_Autorec.history, file_pi)

# # convert the history.history dict to a pandas DataFrame:
# hist_df = pd.DataFrame(hist_Autorec.history)
#
# # save to json:
# hist_json_file = 'history-512-1-sl-001.json'
# with open(hist_json_file, mode='w') as f:
#     hist_df.to_json(f)
#
# # or save to csv:
# hist_csv_file = 'history-512-1-sl-001.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
#
#
# show_rmse(hist_Autorec, 30)
#
# show_error(hist_Autorec, 50)
#
# test_result = AutoRec.evaluate(users_items_matrix_train_three.T, users_items_matrix_test.T)
#
# # print('Test results: loss: '+ str(test_result[0]) + ' - rmse: '+ str(test_result[1]))
#
# predict_autorec = AutoRec.predict(users_items_matrix_train_zero.T)


