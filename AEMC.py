from collections import namedtuple

import tensorflow as tf
import tensorflow_probability as tfp
import scipy
import numpy as np
import pandas as pd

from scipy import optimize

class PrePossData():

    def __init__(self, data, normalizer, miss_entry, miss_rate):
        """ Pre-processing of data.

        Parameters
        ----------
        data : array matrix
            data used
        normalizer : float
            normalize the data
        miss_entrie : float
            number which correspond to the missing entries
        miss_rate : float
            number betwenn (0.0,1.0) which is used to determine the number of
            observed entries. 0.0 => true data - 1.0 => no data
        """
        self.data = tf.convert_to_tensor(data, dtype = tf.float32)
        self.normalizer = tf.convert_to_tensor(normalizer)
        self.miss_entry = tf.convert_to_tensor(miss_entry)
        self.miss_rate = tf.convert_to_tensor(miss_rate)
        self.normalize()
        self.observed_entries()

    def normalize(self):
        """Normalize the data."""
        self.x = tf.divide(self.data,self.normalizer)

    def hadam_matrix(self):
        """Return a matrix of 1 and 0.

        The 1 correspond to the observed entries of the data.
        """
        hadam = tf.ones(self.x.shape)
        hadam = tf.where(self.x > self.miss_entry,
                         x = hadam, y = 0.0)
        return hadam

    def observed_entries(self):
        """ Takes some of the true observed entries of the data.

        The rest of the data equal to zero.
        """
        self.m_original = self.hadam_matrix()
        m_transition = np.ones(self.x.shape)
        for i in range(self.x.shape[0]):
            idx = np.where(self.m_original[i] != 0.0)
            len_idx = idx[0].size
            rand_idx = np.random.permutation(idx[0])
            len_miss = int(np.ceil(len_idx * self.miss_rate))
            miss_idx = rand_idx[0:len_miss]
            m_transition[i][miss_idx] = 0.0

        m_transition = tf.convert_to_tensor(m_transition, dtype = tf.float32)
        self.psi = tf.multiply(m_transition, self.m_original)
        self.xo = tf.multiply(self.x, self.psi)

    def parameters(self):
        """ Some info about the parameters of the pre processing data."""
        index_one = np.random.randint(self.x.shape[0])
        index_two = np.random.randint(self.x.shape[0])

        print("Data size: (" + str(self.x.shape[0]) + "," + str(self.x.shape[1]) + ")")
        print("Psi size: (" + str(self.psi.shape[0]) + "," + str(self.psi.shape[1]) + ")")
        print("Turned data size: (" + str(self.xo.shape[0]) + "," + str(self.xo.shape[1]) + ")")

        print("One row of X: " + str(self.x[index_one][0:20]))
        print("One true parameter: " + str(self.m_original[index_one][0:20]))
        print("One row of Psi: " + str(self.psi[index_one][0:20]))
        print("One row of X used: " + str(self.xo[index_one][0:20]))
        print("Row number: " + str(index_one))

        print("Another row of X: " + str(self.x[index_two][0:20]))
        print("Another true parameter: " + str(self.m_original[index_two][0:20]))
        print("Another row of Psi: " + str(self.psi[index_two][0:20]))
        print("Another row of x used: " + str(self.xo[index_two][0:20]))
        print("Row number: " + str(index_two))


class NeuralNetwork():
    """Neural network's setups."""

    def __init__(self, layer_sizes, act_func,
                 psi,
                 learning_rate = 1.0,
                 weight_decay_l2 = 0.0):
        """Neural network's initializer.

        Parameters
        ----------
        layer_sizes : array(int)
            Size of each layer
        act_func : array(string)
            Activation function for each layer
        psi : array(int)
            Psi matrix of the observed sample
        learning_rate : float32
            Number between 0.0 (No weight's change) and 1.0 (Full weight's change)
        weight_decay_l2 : float32
            Influence of the sum square weight in the objective function
        """
        self.layer_sizes = layer_sizes
        self.act_func = act_func
        self.psi = psi
        self.learning_rate = tf.convert_to_tensor(learning_rate)
        self.weight_decay_l2 = tf.convert_to_tensor(weight_decay_l2)
        # Number of layers
        self.layer = len(self.layer_sizes)
        # Weight matrices between each layer
        self.w = self.weight_initializer()
        # Entry matrices between each layer
        self.a = np.array([None] * (self.layer))
        # Error
        self.e = [None]
        # Loss
        self.loss = 0.0
        # Derivative parameters
        self.dw = [None] * (self.layer - 1)
        # Derivative samples
        self.dx = [None]

    def weight_initializer(self):
        """Weight initializer with a random distribution between -0.5 and 0.5.

        Return
        ------
        w : array(float)
            The weight matrices stored in an array
        """
        w = np.array([None] * (self.layer - 1))
        for i in range(1, self.layer):
            # The bias is added (This is why the "+ 1")
            shape = (self.layer_sizes[i],
                     self.layer_sizes[i - 1] + 1)

            rand = np.random.rand(shape[0],shape[1])-0.5
            rand = 2.0 * 4.0 * rand
            s = np.sqrt(6.0 / (shape[0] + shape[1] - 1))
            rand = rand * s
            w[i - 1] = tf.convert_to_tensor(rand, dtype = tf.float32)
        return w

    def set_entry(self, new_entry, index):
        """Set the entry at index.

        Parameters
        ----------
        new_entry : array(float)
            New entry matrix
        index : int
            Index of the entry matrix
        """
        self.a[index] = new_entry

def feed_forward(neural_network, x_input, x_true):
    """Feed forward inside the neural network.

    The neural network is an object with the activation functions and the weight
    matrices between each layer. The feed forward function compute the error and
    the loss for the forward pass.

    Parameters
    ----------
    neural_network = obj(NeuralNetwork)
        The neural network considered
    x_input : array(float)
        The input matrix
    x_true : array(float)
        The matrix for error computation with the output layer
    """
    samples = x_input.shape[0]
    # Adding manually the bias
    bias = tf.ones((samples,1))
    x_bias = tf.concat([bias,x_input],-1)
    # Number of layers
    layer = neural_network.layer
    # Set input entry
    neural_network.set_entry(x_bias, 0)

    # Loop to set new entries
    for i in range(1, layer - 1):
        lin_act = tf.matmul(neural_network.a[i - 1],
                            neural_network.w[i - 1],
                            transpose_b=True)
        if (neural_network.act_func[i - 1] == 'tanh'):
            new_entry = 1.7159*tf.math.tanh(2.0/3.0*lin_act)
        else:
            new_entry = tf.math.sigmoid(lin_act)
        # Adding manually the bias
        new_entry = tf.concat([bias,new_entry],-1)
        neural_network.set_entry(new_entry, i)

    # Output
    lin_act = tf.matmul(neural_network.a[layer - 2],
                        neural_network.w[layer - 2],
                        transpose_b = True)
    if (neural_network.act_func[layer - 2] == 'tanh'):
        new_entry = 1.7159*tf.math.tanh(2.0/3.0*lin_act)
    else:
        new_entry = tf.math.sigmoid(lin_act)
    neural_network.set_entry(new_entry, layer - 1)

    # Error computation
    neural_network.e = x_true - neural_network.a[layer - 1]
    # Take only the error of the observed entry
    neural_network.e = tf.multiply(neural_network.e, neural_network.psi)

    # Loss computation
    neural_network.loss = tf.reduce_sum(tf.math.square(neural_network.e)) / (2.0 * samples)

def back_propagation(neural_network):
    """ Computation of the partial derivatives of the objective function J(X,theta) = L(X,theta) + Gamma(Theta).

    This derivates are computed with respect to X and theta. dL/dtheta is done thanks to back propagation
    algorithm and dL/dX is done thanks to the first hidden layer and the first weight matrix.

    Parameters:
    -----------
    neural_network = obj(NeuralNetwork)
        The neural network considered
    """
    # Number of layers
    layer = neural_network.layer
    # Partial derivatives via back propagation of each layer
    d = [None] * layer

    # Back propagation outer state
    if (neural_network.act_func[layer - 2] == 'tanh'):
        # Derivative of tanh with the last state (output layer) and the error
        derivative = (1.7159 * 2.0/3.0 * np.square(1.0 - 1.0/1.7159) * tf.square(neural_network.a[layer - 1]))
    else:
        # Derivative of sigmoid with the last state (output layer) and the error
        derivative = tf.multiply(neural_network.a[layer - 1],
                                 1 - neural_network.a[layer - 1])
    d[layer - 1] = tf.multiply(-neural_network.e, derivative)

    # Loop over the inner states
    for i in range(layer - 2, 0, -1):
        if (neural_network.act_func[i - 1] == 'tanh'):
            derivative = (1.7159 * 2.0/3.0 * np.square(1.0 - 1.0/1.7159) * tf.square(neural_network.a[i]))
        else:
            derivative = tf.multiply(neural_network.a[i],
                                     1 - neural_network.a[i])

        if (i + 1 == layer - 1):
            # Do not remove any bias
            d[i] = tf.multiply(tf.matmul(d[i + 1], neural_network.w[i]), derivative)
        else:
            # Remove bias
            d[i] = tf.multiply(tf.matmul(d[i + 1][:, 1:], neural_network.w[i]), derivative)

    # Back propagation algorithm
    for i in range(layer - 1):
        if (i + 1 == layer - 1):
            neural_network.dw[i] = tf.matmul(d[i + 1],
                                             neural_network.a[i],
                                             transpose_a=True) / d[i + 1].shape[0]
        else:
            neural_network.dw[i] = tf.matmul(d[i + 1][:, 1:],
                                             neural_network.a[i],
                                             transpose_a=True) / d[i + 1].shape[0]

    # Partial derivatives of objective, function of X
    neural_network.dx = neural_network.e + tf.matmul(d[1][:, 1:],
                                                     neural_network.w[0][:, 1:])
    neural_network.dx = neural_network.dx / neural_network.a[0].shape[0]
    not_psi = (neural_network.psi + 1) % 2
    neural_network.dx = tf.multiply(neural_network.dx, not_psi)

def aemc_propagation_f(neural_network,x_input):
    """ In comment ... """
    # Take the transition state at the input layer without the bias
    x_transition = neural_network.a[0][:,1:].numpy()
    # Find the unobserved data thanks to the hadamrd matrix in the input matrix
    index = np.where(neural_network.psi.numpy() == 0)
    len_missing = len(index[0])
    # Replace it in the transition matrix state
    # state = x_input[:len_missing,0].numpy()
    state = x_input[:len_missing]
    x_transition[index] = state
    x_transition = tf.convert_to_tensor(x_transition)
    t = len_missing
    for i in range(len(neural_network.w)):
        (a,b) = neural_network.w[i].shape
        neural_network.w[i] = tf.transpose(tf.reshape(x_input[t:t + a * b],[b,a]))
        t = t + a * b
    # Feed-Forward pass the transition state
    feed_forward(neural_network,x_transition,x_transition)
    back_propagation(neural_network)
    # Gradient descent and minimization function
    l2_w = 0.0
    for i in range(len(neural_network.w)):
        l2_w += tf.reduce_sum(tf.square(neural_network.w[i]))
    f = neural_network.loss + 0.5 * neural_network.weight_decay_l2 * l2_w
    return f

def aemc_propagation_g(neural_network):
    # Back-Propagation pass
    back_propagation(neural_network)
    g = []
    index = np.where(neural_network.psi == 0)
    for i in range(len(neural_network.w)):
      dw = tf.reshape(tf.transpose(neural_network.dw[i]),[-1])
      w = tf.reshape(tf.transpose(neural_network.w[i]),[-1])
      g = np.r_[g, dw + neural_network.weight_decay_l2 * w]
    gx = neural_network.dx.numpy()
    gx = gx[index]
    g = np.r_[gx,g]
    return g

def aemc_propagation_f_bis(neural_network):
    def func(x_input):
        """ In comment ... """
        # Take the transition state at the input layer without the bias
        x_transition = neural_network.a[0][:, 1:].numpy()
        # Find the unobserved data thanks to the hadamrd matrix in the input matrix
        index = np.where(neural_network.psi.numpy() == 0)
        len_missing = len(index[0])
        # Replace it in the transition matrix state
        # state = x_input[:len_missing, 0].numpy()
        state = x_input[:len_missing]
        x_transition[index] = state
        x_transition = tf.convert_to_tensor(x_transition)
        t = len_missing
        for i in range(len(neural_network.w)):
            (a, b) = neural_network.w[i].shape
            neural_network.w[i] = tf.cast(tf.transpose(tf.reshape(x_input[t:t + a * b], [b, a])),tf.float32)
            t = t + a * b
        # Feed-Forward pass the transition state
        feed_forward(neural_network, x_transition, x_transition)
        back_propagation(neural_network)
        # Gradient descent and minimization function
        l2_w = 0.0
        for i in range(len(neural_network.w)):
            l2_w += tf.reduce_sum(tf.square(neural_network.w[i]))
        f = neural_network.loss + 0.5 * neural_network.weight_decay_l2 * l2_w
        return f.numpy()
    return func

def aemc_propagation_g_bis(neural_network):
    def func(x_input):
        # Back-Propagation pass)
        back_propagation(neural_network)
        g = []
        index = np.where(neural_network.psi == 0)
        for i in range(len(neural_network.w)):
            dw = tf.reshape(tf.transpose(neural_network.dw[i]), [-1])
            w = tf.reshape(tf.transpose(neural_network.w[i]), [-1])
            g = np.r_[g, dw + neural_network.weight_decay_l2 * w]
        gx = neural_network.dx.numpy()
        gx = gx[index]
        g = np.r_[gx, g]
        return g
    return func


def aemc_optimizer(neural_network, x_input, max_iter,tol_fun):
    """AutoEncoder based matrix completion optimizer.

    Computation of the feed-forward and the back-propagation with a minimization
    objective function based on conjugate gradient descent.

    Parameters
    ----------
    neural_network = obj(NeuralNetwork)
        The neural network considered
    x_input : array(float)
        The input matrix
    """
    # feed-forward pass
    feed_forward(neural_network, x_input, x_input)
    w = []
    for i in range(len(neural_network.w)):
        w = np.r_[w, tf.reshape(tf.transpose(neural_network.w[i]), [-1])]
    index = np.where(neural_network.psi == 0)
    x_missing = x_input[index]
    y = np.r_[x_missing, w]

    x = y
    f = aemc_propagation_f(neural_network,x)
    g = aemc_propagation_g(neural_network)

    function_evaluations = 1

    for i in range(max_iter):
        print(i)
        if i == 0:
            # Initially use steepest direction
            d = -g
        else:
            g_tran_g_old = np.matmul(g.T,g_old)
            g_old_tran_g_old = np.matmul(g_old.T,g_old)
            # Hestenes-Stiefel update
            beta = np.matmul(g.T,(g-g_old))/np.matmul((g-g_old).T,d)
            d = -g * beta*d

            # Restart if not a direction of sufficient descent
            if np.matmul(g.T,d) > -1.0e-9:
                beta = 0
                d = -g

        g_old = g

        g_tran_d = np.matmul(g.T,d)
        if g_tran_d > - 1.0e-9:
            exitflag = 2
            break

        # Select initial guess
        if i == 0:
            t = np.minimum(1,1/np.sum(np.abs(g)))
        else:
            # Quadratic initialization based on {f,g} and previous f
            t = np.minimum(1,2*(f-f_old)/g_tran_d)

        f_old = f
        g_tran_d_old = g_tran_d
        fr = f

        results = optimize.line_search(aemc_propagation_f_bis(neural_network),      # Objective function.
                                       aemc_propagation_g_bis(neural_network),      # Objective function gradient.
                                       x,                                           # Starting point.
                                       d,                                           # Search direction.
                                       gfk = g,
                                       old_fval = f_old,
                                       c1 = 1.0e-4,                                 # Armijo condition rule
                                       c2 = 0.2,                                    # Curvature condition rule
                                       maxiter = 25)                                # Maximum number of iteration to perform
        function_evaluations = function_evaluations + results[1]
        if results[0] == None:
            x = x + t*d
        else :
            t = results[0]
            x = x + results[0] * d

        if results[3] == None:
            f = results[4]
        else:
            f = results[3]

        if np.all(results[5] == None):
            g = g
        else:
            g = results[5]

        if np.sum(np.abs(g)) <= tol_fun:
            exitflag = 1
            msg = 'Optimality condition below the tolerance function'
            break;

        if np.sum(np.abs(t*d))  <= 1.0e-9:
            exitflag = 2
            msg = 'Step Size below TolX'
            break

        if (f-f_old) < 1.0-9:
            exitflag = 2
            msg = 'Function Value changing by less than TolX'
            break

        if function_evaluations > 2*max_iter:
            exitflag = 0
            msg = 'Exceeded Maximum Number of Function Evaluations'
            break

        if i == max_iter:
            exitflag = 0
            msg = 'Exceeded Maximum Number of Iterations'
            break

    print(msg)

    return x, f, exitflag

def aemc_optimizer2(neural_network, x_input, max_iter,tol_fun):
    """AutoEncoder based matrix completion optimizer.

    Computation of the feed-forward and the back-propagation with a minimization
    objective function based on conjugate gradient descent.

    Parameters
    ----------
    neural_network = obj(NeuralNetwork)
        The neural network considered
    x_input : array(float)
        The input matrix
    """
    # feed-forward pass
    feed_forward(neural_network, x_input, x_input)
    w = None
    for i in range(len(neural_network.w)):
        if w == None:
            w = tf.reshape(tf.transpose(neural_network.w[i]), [-1, 1])
        else:
            w = tf.concat([w, tf.reshape(tf.transpose(neural_network.w[i]), [-1, 1])],0)
    index = np.where(neural_network.psi.numpy() == 0)
    x_missing = x_input.numpy()[index]
    len_missing = len(x_missing)
    y = np.r_[x_missing, w[:,0].numpy()]
    y = y.astype('float32')
    aemc_propagation_g(neural_network)
    results = scipy.optimize.fmin_cg(aemc_propagation_f_bis(neural_network),
                                     y,
                                    fprime = aemc_propagation_g_bis(neural_network),
                                    maxiter = max_iter)
    x_exit = x_input.numpy()
    x_exit[index] = results[:len_missing]
    t = len_missing
    for i in range(len(neural_network.w)):
        (a, b) = neural_network.w[i].shape
        neural_network.w[i] = tf.cast(tf.transpose(tf.reshape(results[t:t + a * b], [b, a])), tf.float32)
        t = t + a * b
    x_exit = tf.convert_to_tensor(x_exit.astype('float32'))
    feed_forward(neural_network,x_exit,x_exit)

if __name__ == '__main__':
    # Define the data in the correct form to train the neural network
    # ---------------------------------------------------------------
    data = pd.read_csv('movielen100k.csv')
    data = data.to_numpy()
    pre_processing_data = PrePossData(data.T, 5.0, 0.0, 0.3)

    x = pre_processing_data.x
    m = pre_processing_data.m_original

    psi = pre_processing_data.psi
    xo = pre_processing_data.xo

    # Display parameters
    # pre_processing_data.parameters()
    # ----------------------------------------------------------------

    # Setup of the neural network
    # ----------------------------------------------------------------
    # Layer sizes
    original_dim = x.shape[1]
    hidden_layer = np.array([50])
    layer_sizes = np.array([original_dim])
    layer_sizes = np.append(layer_sizes, hidden_layer)
    layer_sizes = np.append(layer_sizes, original_dim)
    # Activation functions
    act_func = ['tanh', 'sigmoid']
    # Computation parameters
    learning_rate = 1.0
    weight_decay_l2 = 0.001
    # Neural network onject
    neural_network = NeuralNetwork(layer_sizes, act_func, psi,
                                   learning_rate=learning_rate,
                                   weight_decay_l2=weight_decay_l2)
    # ----------------------------------------------------------------

    # Training AEMC
    # ----------------------------------------------------------------
    max_iter = 500
    tol_fun = 1.0e-5

    aemc_optimizer2(neural_network, xo, max_iter,tol_fun)
    not_psi = (psi+1)%2
    x_new = tf.multiply(xo,psi)+tf.multiply(neural_network.a[-1],not_psi)
    mm = tf.multiply(not_psi,m)
    e = tf.multiply((x_new-x),mm)
    nmae = tf.reduce_sum(tf.abs(e))/tf.reduce_sum(mm)
    x_new = x_new.numpy()
    print("Application finished")
