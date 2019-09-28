import numpy as np
import keras.backend as K
from keras.layers import *
from keras.initializers import *
from keras.models import *
import random
from keras.utils.vis_utils import plot_model
import math

# fixing sedd value: uniform reporting of results.
from numpy.random import seed
seed(16)
from tensorflow import set_random_seed
set_random_seed(16)

##############
# NALU Class #
##############

GR = 1.61803399

class NALU(Layer):
    def __init__(self, units, MW_initializer='glorot_uniform',
                 G_initializer='glorot_uniform', mode="NALU",
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.MW_initializer = initializers.get(MW_initializer)
        self.G_initializer = initializers.get(G_initializer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='W_hat')
        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.MW_initializer,
                                     name='M_hat')

        if self.mode == "NALU":
            self.G = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.G_initializer,
                                     name='G')
                                     
        if self.mode == "GNALU":
            self.G = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.G_initializer,
                                     name='G')
                                     
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
    
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        a = K.dot(inputs, W)
        
        if self.mode == "GNAC" or self.mode == "GNALU" :
            # gold_tanh & gold_sigmoid are defined in keras.backend source code file.
            W = K.gold_tanh(self.W_hat) * K.gold_sigmoid(self.M_hat)
            a = K.dot(inputs, W)
            
        if self.mode == "NAC":
            output = a
        elif self.mode == "GNAC":
            output = a
        elif self.mode == "NALU":
            m = K.exp(K.dot(K.log(K.abs(inputs) + 1e-7), W))
            g = K.sigmoid(K.dot(K.abs(inputs), self.G))
            output = g * a + (1 - g) * m
        elif self.mode == "GNALU":
            m = K.pow(GR, (K.dot(K.log(K.abs(inputs) + 1e-7)/K.log(GR), W)))
            g = K.gold_sigmoid(K.dot(K.abs(inputs), self.G))
            output = g * a + (1 - g) * m
        else:
            raise ValueError("Valid modes: 'NAC', 'NALU', 'GNAC', 'GNALU'.")
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'mode' : self.mode,
            'MW_initializer': initializers.serialize(self.MW_initializer),
            'G_initializer':  initializers.serialize(self.G_initializer)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


####################
# NALU based model #
####################

def nalu_model(mode='NALU', inp_val=2):

    # inp_val as per our experimental setup can have values 1 or 2.
    x = Input((inp_val,))
    y = NALU(3, mode=mode, 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(x)

    y = NALU(1, mode=mode, 
             MW_initializer=RandomNormal(stddev=1),
             G_initializer=Constant(10))(y)
    
    return Model(x, y)


#############
# MLP model #
#############

def mlp_model(inp_val):

    # inp_val as per our experimental setup can have values 1 or 2.
    x = Input((inp_val,))
    y = Dense(3, activation="relu")(x)
    y = Dense(1)(y)
    
    return Model(x, y)


def get_bin_data(N, op):
    split = 1
    trX = np.random.uniform(-10, 10, (N, 2))
    a = trX[:, :split].sum(1)
    b = trX[:, split:].sum(1)
    print(a.min(), a.max(), b.min(), b.max())
    trY = op(a, b)[:, None]
    teX = np.random.uniform(-50,50, (N, 2)) 
    a = teX[:, :split].sum(1)
    b = teX[:, split:].sum(1)
    print(a.min(), a.max(), b.min(), b.max())
    teY = op(a, b)[:, None]
    return (trX, trY), (teX, teY)


def get_un_data(N, op):
    trX = np.random.uniform(0, 10, (N, 1))
    print(trX.min(), trX.max())
    trY = op(trX)
    teX = np.random.uniform(0,50, (N, 1)) 
    print(teX.min(), teX.max())
    teY = op(teX)
    return (trX, trY), (teX, teY)


def get_sqrt_data(N):
    trX = np.random.uniform(0, 10, (N, 1))
    print(trX.min(), trX.max())
    trY = np.sqrt(trX)
    teX = np.random.uniform(0,50, (N, 1)) 
    print(teX.min(), teX.max())
    teY = np.sqrt(teX)
    return (trX, trY), (teX, teY)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# generate data: same data generated for all models & with same random seed.
(trx_add, try_add), (tex_add, tey_add) = get_bin_data(2 ** 16, lambda a, b: a + b)
(trx_sub, try_sub), (tex_sub, tey_sub) = get_bin_data(2 ** 16, lambda a, b: a - b)
(trx_mul, try_mul), (tex_mul, tey_mul) = get_bin_data(2 ** 16, lambda a, b: a * b)
(trx_div, try_div), (tex_div, tey_div) = get_bin_data(2 ** 16, lambda a, b: a / b)
(trx_sqr, try_sqr), (tex_sqr, tey_sqr) = get_un_data(2 ** 16, lambda a: a * a)
(trx_qrt, try_qrt), (tex_qrt, tey_qrt) = get_sqrt_data(2 ** 16)

def mse_trainer_mlp():
    
    bin_val = 2
    un_val = 1
    # training add mlp and storing it's history values for plotting.
    m_add = mlp_model(bin_val)
    m_add.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_add.optimizer.lr, 1e-2)
    hist_m_add_u = m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    K.set_value(m_add.optimizer.lr, 1e-3)
    hist_m_add_d =  m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    K.set_value(m_add.optimizer.lr, 1e-4)
    hist_m_add_t = m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    
    m_add.save('mlp_mse_add.h5')
    
    # training sub mlp and storing it's history values for plotting.
    m_sub = mlp_model(bin_val)
    m_sub.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_sub.optimizer.lr, 1e-2)
    hist_m_sub_u = m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    K.set_value(m_sub.optimizer.lr, 1e-3)
    hist_m_sub_d =  m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    K.set_value(m_sub.optimizer.lr, 1e-4)
    hist_m_sub_t = m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    
    m_sub.save('mlp_mse_sub.h5')

    # training mul mlp and storing it's history values for plotting.
    m_mul = mlp_model(bin_val)
    m_mul.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_mul.optimizer.lr, 1e-2)
    hist_m_mul_u = m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)
    K.set_value(m_mul.optimizer.lr, 1e-3)
    hist_m_mul_d =  m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)
    K.set_value(m_mul.optimizer.lr, 1e-4)
    hist_m_mul_t = m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)    
    
    m_mul.save('mlp_mse_mul.h5')
    
    # training div mlp and storing it's history values for plotting.
    m_div = mlp_model(bin_val)
    m_div.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_div.optimizer.lr, 1e-2)
    hist_m_div_u = m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)
    K.set_value(m_div.optimizer.lr, 1e-3)
    hist_m_div_d =  m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)
    K.set_value(m_div.optimizer.lr, 1e-4)
    hist_m_div_t = m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)    
    
    m_div.save('mlp_mse_div.h5')
    
    # training sqr mlp and storing it's history values for plotting.
    m_sqr = mlp_model(un_val)
    m_sqr.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_sqr.optimizer.lr, 1e-2)
    hist_m_sqr_u = m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)
    K.set_value(m_sqr.optimizer.lr, 1e-3)
    hist_m_sqr_d =  m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)
    K.set_value(m_sqr.optimizer.lr, 1e-4)
    hist_m_sqr_t = m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)
    
    m_sqr.save('mlp_mse_sqr.h5')

    # training qrt mlp and storing it's history values for plotting.
    m_qrt = mlp_model(un_val)
    m_qrt.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_qrt.optimizer.lr, 1e-2)
    hist_m_qrt_u = m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)
    K.set_value(m_qrt.optimizer.lr, 1e-3)
    hist_m_qrt_d =  m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)
    K.set_value(m_qrt.optimizer.lr, 1e-4)
    hist_m_qrt_t = m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)
    
    m_qrt.save('mlp_mse_qrt.h5')

    return hist_m_add_u, hist_m_add_d, hist_m_add_t, hist_m_sub_u, hist_m_sub_d, hist_m_sub_t, \
           hist_m_mul_u, hist_m_mul_d, hist_m_mul_t, hist_m_div_u, hist_m_div_d, hist_m_div_t, \
           hist_m_sqr_u, hist_m_sqr_d, hist_m_sqr_t, hist_m_qrt_u, hist_m_qrt_d, hist_m_qrt_t
    

def rmse_trainer_mlp():

    bin_val = 2
    un_val = 1
    # training add mlp and storing it's history values for plotting.
    m_add = mlp_model(bin_val)
    m_add.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_add.optimizer.lr, 1e-2)
    hist_m_add_u = m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    K.set_value(m_add.optimizer.lr, 1e-3)
    hist_m_add_d =  m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    K.set_value(m_add.optimizer.lr, 1e-4)
    hist_m_add_t = m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    
    m_add.save('mlp_rmse_add.h5')
    
    # training sub mlp and storing it's history values for plotting.
    m_sub = mlp_model(bin_val)
    m_sub.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_sub.optimizer.lr, 1e-2)
    hist_m_sub_u = m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    K.set_value(m_sub.optimizer.lr, 1e-3)
    hist_m_sub_d =  m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    K.set_value(m_sub.optimizer.lr, 1e-4)
    hist_m_sub_t = m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    
    m_sub.save('mlp_rmse_sub.h5')

    # training mul mlp and storing it's history values for plotting.
    m_mul = mlp_model(bin_val)
    m_mul.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_mul.optimizer.lr, 1e-2)
    hist_m_mul_u = m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)
    K.set_value(m_mul.optimizer.lr, 1e-3)
    hist_m_mul_d =  m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)
    K.set_value(m_mul.optimizer.lr, 1e-4)
    hist_m_mul_t = m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)    
    
    m_mul.save('mlp_rmse_mul.h5')
    
    # training div mlp and storing it's history values for plotting.
    m_div = mlp_model(bin_val)
    m_div.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_div.optimizer.lr, 1e-2)
    hist_m_div_u = m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)
    K.set_value(m_div.optimizer.lr, 1e-3)
    hist_m_div_d =  m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)
    K.set_value(m_div.optimizer.lr, 1e-4)
    hist_m_div_t = m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)    
    
    m_div.save('mlp_rmse_div.h5')
    
    # training sqr mlp and storing it's history values for plotting.
    m_sqr = mlp_model(un_val)
    m_sqr.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_sqr.optimizer.lr, 1e-2)
    hist_m_sqr_u = m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)
    K.set_value(m_sqr.optimizer.lr, 1e-3)
    hist_m_sqr_d =  m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)
    K.set_value(m_sqr.optimizer.lr, 1e-4)
    hist_m_sqr_t = m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)    
    
    m_sqr.save('mlp_rmse_sqr.h5')
    
    # training qrt mlp and storing it's history values for plotting.
    m_qrt = mlp_model(un_val)
    m_qrt.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_qrt.optimizer.lr, 1e-2)
    hist_m_qrt_u = m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)
    K.set_value(m_qrt.optimizer.lr, 1e-3)
    hist_m_qrt_d =  m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)
    K.set_value(m_qrt.optimizer.lr, 1e-4)
    hist_m_qrt_t = m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)    
    
    m_qrt.save('mlp_rmse_sqr.h5')
    
    return hist_m_add_u, hist_m_add_d, hist_m_add_t, hist_m_sub_u, hist_m_sub_d, hist_m_sub_t, \
           hist_m_mul_u, hist_m_mul_d, hist_m_mul_t, hist_m_div_u, hist_m_div_d, hist_m_div_t, \
           hist_m_sqr_u, hist_m_sqr_d, hist_m_sqr_t, hist_m_qrt_u, hist_m_qrt_d, hist_m_qrt_t    

def mse_trainer_nalu(model_type):
    
    bin_val = 2
    un_val = 1
    # training add nalu/nac and storing it's history values for plotting.
    m_add = nalu_model(model_type, bin_val)
    m_add.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_add.optimizer.lr, 1e-2)
    hist_m_add_u = m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    K.set_value(m_add.optimizer.lr, 1e-3)
    hist_m_add_d =  m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    K.set_value(m_add.optimizer.lr, 1e-4)
    hist_m_add_t = m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    
    
    if model_type == 'NAC':
        m_add.save_weights('nac_mse_add.h5')
    elif model_type == 'GNAC':
        m_add.save_weights('gnac_mse_add.h5')
    elif model_type == 'NALU':
        m_add.save_weights('nalu_mse_add.h5')
    elif model_type == 'GNALU':
        m_add.save_weights('gnalu_mse_add.h5')
    
    
    # training sub nalu/nac and storing it's history values for plotting.
    m_sub = nalu_model(model_type, bin_val)
    m_sub.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_sub.optimizer.lr, 1e-2)
    hist_m_sub_u = m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    K.set_value(m_sub.optimizer.lr, 1e-3)
    hist_m_sub_d =  m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    K.set_value(m_sub.optimizer.lr, 1e-4)
    hist_m_sub_t = m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_sub.save_weights('nac_mse_sub.h5')
    elif model_type == 'GNAC':
        m_sub.save_weights('gnac_mse_sub.h5')
    elif model_type == 'NALU':
        m_sub.save_weights('nalu_mse_sub.h5')
    elif model_type == 'GNALU':
        m_sub.save_weights('gnalu_mse_sub.h5')


    # training mul nalu/nac and storing it's history values for plotting.
    m_mul = nalu_model(model_type, bin_val)
    m_mul.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_mul.optimizer.lr, 1e-2)
    hist_m_mul_u = m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)
    K.set_value(m_mul.optimizer.lr, 1e-3)
    hist_m_mul_d =  m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)
    K.set_value(m_mul.optimizer.lr, 1e-4)
    hist_m_mul_t = m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_mul.save_weights('nac_mse_mul.h5')
    elif model_type == 'GNAC':
        m_mul.save_weights('gnac_mse_mul.h5')
    elif model_type == 'NALU':
        m_mul.save_weights('nalu_mse_mul.h5')
    elif model_type == 'GNALU':
        m_mul.save_weights('gnalu_mse_mul.h5')
  
    # training div nalu/nac and storing it's history values for plotting.
    m_div = nalu_model(model_type, bin_val)
    m_div.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_div.optimizer.lr, 1e-2)
    hist_m_div_u = m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)
    K.set_value(m_div.optimizer.lr, 1e-3)
    hist_m_div_d =  m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)
    K.set_value(m_div.optimizer.lr, 1e-4)
    hist_m_div_t = m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_div.save_weights('nac_mse_div.h5')
    elif model_type == 'GNAC':
        m_div.save_weights('gnac_mse_div.h5')
    elif model_type == 'NALU':
        m_div.save_weights('nalu_mse_div.h5')
    elif model_type == 'GNALU':
        m_div.save_weights('gnalu_mse_div.h5')

    # training sqr nalu/nac and storing it's history values for plotting.
    m_sqr = nalu_model(model_type, un_val)
    m_sqr.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_sqr.optimizer.lr, 1e-2)
    hist_m_sqr_u = m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)
    K.set_value(m_sqr.optimizer.lr, 1e-3)
    hist_m_sqr_d =  m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)
    K.set_value(m_sqr.optimizer.lr, 1e-4)
    hist_m_sqr_t = m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_sqr.save_weights('nac_mse_sqr.h5')
    elif model_type == 'GNAC':
        m_sqr.save_weights('gnac_mse_sqr.h5')
    elif model_type == 'NALU':
        m_sqr.save_weights('nalu_mse_sqr.h5')
    elif model_type == 'GNALU':
        m_sqr.save_weights('gnalu_mse_sqr.h5')

    # training qrt nalu/nac and storing it's history values for plotting.
    m_qrt = nalu_model(model_type, un_val)
    m_qrt.compile("nadam", "mse", metrics=["mae", rmse])
    K.set_value(m_qrt.optimizer.lr, 1e-2)
    hist_m_qrt_u = m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)
    K.set_value(m_qrt.optimizer.lr, 1e-3)
    hist_m_qrt_d =  m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)
    K.set_value(m_qrt.optimizer.lr, 1e-4)
    hist_m_qrt_t = m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_qrt.save_weights('nac_mse_qrt.h5')
    elif model_type == 'GNAC':
        m_qrt.save_weights('gnac_mse_qrt.h5')
    elif model_type == 'NALU':
        m_qrt.save_weights('nalu_mse_qrt.h5')
    elif model_type == 'GNALU':
        m_qrt.save_weights('gnalu_mse_qrt.h5')

    return hist_m_add_u, hist_m_add_d, hist_m_add_t, hist_m_sub_u, hist_m_sub_d, hist_m_sub_t, \
           hist_m_mul_u, hist_m_mul_d, hist_m_mul_t, hist_m_div_u, hist_m_div_d, hist_m_div_t, \
           hist_m_sqr_u, hist_m_sqr_d, hist_m_sqr_t, hist_m_qrt_u, hist_m_qrt_d, hist_m_qrt_t
    

def rmse_trainer_nalu(model_type):

    bin_val = 2
    un_val = 1
    # training add nalu/nac and storing it's history values for plotting.
    m_add = nalu_model(model_type, bin_val)
    m_add.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_add.optimizer.lr, 1e-2)
    hist_m_add_u = m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    K.set_value(m_add.optimizer.lr, 1e-3)
    hist_m_add_d =  m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)
    K.set_value(m_add.optimizer.lr, 1e-4)
    hist_m_add_t = m_add.fit(trx_add, try_add, validation_data=(tex_add, tey_add), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_add.save_weights('nac_rmse_add.h5')
    elif model_type == 'GNAC':
        m_add.save_weights('gnac_rmse_add.h5')
    elif model_type == 'NALU':
        m_add.save_weights('nalu_rmse_add.h5')
    elif model_type == 'GNALU':
        m_add.save_weights('gnalu_rmse_add.h5')
    
    # training sub nalu/nac and storing it's history values for plotting.
    m_sub = nalu_model(model_type, bin_val)
    m_sub.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_sub.optimizer.lr, 1e-2)
    hist_m_sub_u = m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    K.set_value(m_sub.optimizer.lr, 1e-3)
    hist_m_sub_d =  m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)
    K.set_value(m_sub.optimizer.lr, 1e-4)
    hist_m_sub_t = m_sub.fit(trx_sub, try_sub, validation_data=(tex_sub, tey_sub), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_sub.save_weights('nac_rmse_sub.h5')
    elif model_type == 'GNAC':
        m_sub.save_weights('gnac_rmse_sub.h5')
    elif model_type == 'NALU':
        m_sub.save_weights('nalu_rmse_sub.h5')
    elif model_type == 'GNALU':
        m_sub.save_weights('gnalu_rmse_sub.h5')

    # training mul nalu/nac and storing it's history values for plotting.
    m_mul = nalu_model(model_type, bin_val)
    m_mul.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_mul.optimizer.lr, 1e-2)
    hist_m_mul_u = m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)
    K.set_value(m_mul.optimizer.lr, 1e-3)
    hist_m_mul_d =  m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)
    K.set_value(m_mul.optimizer.lr, 1e-4)
    hist_m_mul_t = m_mul.fit(trx_mul, try_mul, validation_data=(tex_mul, tey_mul), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_mul.save_weights('nac_rmse_mul.h5')
    elif model_type == 'GNAC':
        m_mul.save_weights('gnac_rmse_mul.h5')
    elif model_type == 'NALU':
        m_mul.save_weights('nalu_rmse_mul.h5')
    elif model_type == 'GNALU':
        m_mul.save_weights('gnalu_rmse_mul.h5')

    # training div nalu/nac and storing it's history values for plotting.
    m_div = nalu_model(model_type, bin_val)
    m_div.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_div.optimizer.lr, 1e-2)
    hist_m_div_u = m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)
    K.set_value(m_div.optimizer.lr, 1e-3)
    hist_m_div_d =  m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)
    K.set_value(m_div.optimizer.lr, 1e-4)
    hist_m_div_t = m_div.fit(trx_div, try_div, validation_data=(tex_div, tey_div), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_div.save_weights('nac_rmse_div.h5')
    elif model_type == 'GNAC':
        m_div.save_weights('gnac_rmse_div.h5')
    elif model_type == 'NALU':
        m_div.save_weights('nalu_rmse_div.h5')
    elif model_type == 'GNALU':
        m_div.save_weights('gnalu_rmse_div.h5')

    # training sqr nalu/nac and storing it's history values for plotting.
    m_sqr = nalu_model(model_type, un_val)
    m_sqr.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_sqr.optimizer.lr, 1e-2)
    hist_m_sqr_u = m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)
    K.set_value(m_sqr.optimizer.lr, 1e-3)
    hist_m_sqr_d =  m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)
    K.set_value(m_sqr.optimizer.lr, 1e-4)
    hist_m_sqr_t = m_sqr.fit(trx_sqr, try_sqr, validation_data=(tex_sqr, tey_sqr), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_sqr.save_weights('nac_rmse_sqr.h5')
    elif model_type == 'GNAC':
        m_sqr.save_weights('gnac_rmse_sqr.h5')
    elif model_type == 'NALU':
        m_sqr.save_weights('nalu_rmse_sqr.h5')
    elif model_type == 'GNALU':
        m_sqr.save_weights('gnalu_rmse_sqr.h5')

    # training qrt nalu/nac and storing it's history values for plotting.
    m_qrt = nalu_model(model_type, un_val)
    m_qrt.compile("nadam", rmse, metrics=["mae", rmse])
    K.set_value(m_qrt.optimizer.lr, 1e-2)
    hist_m_qrt_u = m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)
    K.set_value(m_qrt.optimizer.lr, 1e-3)
    hist_m_qrt_d =  m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)
    K.set_value(m_qrt.optimizer.lr, 1e-4)
    hist_m_qrt_t = m_qrt.fit(trx_qrt, try_qrt, validation_data=(tex_qrt, tey_qrt), batch_size=1024, epochs=100)    

    if model_type == 'NAC':
        m_qrt.save_weights('nac_rmse_qrt.h5')
    elif model_type == 'GNAC':
        m_qrt.save_weights('gnac_rmse_qrt.h5')
    elif model_type == 'NALU':
        m_qrt.save_weights('nalu_rmse_qrt.h5')
    elif model_type == 'GNALU':	
        m_qrt.save_weights('gnalu_rmse_qrt.h5')

    return hist_m_add_u, hist_m_add_d, hist_m_add_t, hist_m_sub_u, hist_m_sub_d, hist_m_sub_t, \
           hist_m_mul_u, hist_m_mul_d, hist_m_mul_t, hist_m_div_u, hist_m_div_d, hist_m_div_t, \
           hist_m_sqr_u, hist_m_sqr_d, hist_m_sqr_t, hist_m_qrt_u, hist_m_qrt_d, hist_m_qrt_t    



import pickle

if __name__ == "__main__":

    # training mse mlp model
    mlp_mse_history_sum = mse_trainer_mlp()

    # training rmse mlp model
    mlp_rmse_history_sum = rmse_trainer_mlp()

    # training mse nac model
    nac_mse_history_sum = mse_trainer_nalu('NAC')

    # training rmse nac model
    nac_rmse_history_sum = rmse_trainer_nalu('NAC')

    # training mse gnac model
    gnac_mse_history_sum = mse_trainer_nalu('GNAC')

    # training rmse gnac model
    gnac_rmse_history_sum = rmse_trainer_nalu('GNAC')
    
    # training mse nalu model
    nalu_mse_history_sum = mse_trainer_nalu('NALU')

    # training rmse nalu model
    nalu_rmse_history_sum = rmse_trainer_nalu('NALU')

    # training mse gnalu model
    gnalu_mse_history_sum = mse_trainer_nalu('GNALU')

    # training rmse gnalu model
    gnalu_rmse_history_sum = rmse_trainer_nalu('GNALU')
    
    print('MLP MSE learning history summarized.')
    print('\n')
    print(mlp_mse_history_sum)

    print('MLP RMSE learning history summarized.')
    print('\n')
    print(mlp_rmse_history_sum)

    print('NAC MSE learning history summarized.')
    print('\n')    
    print(nac_mse_history_sum)
    
    print('NAC RMSE learning history summarized.')
    print('\n')    
    print(nac_rmse_history_sum)

    print('GNAC MSE learning history summarized.')
    print('\n')    
    print(gnac_mse_history_sum)

    print('GNAC RMSE learning history summarized.')
    print('\n')    
    print(gnac_rmse_history_sum)

    print('NALU MSE learning history summarized.')
    print('\n')
    print(nalu_mse_history_sum)
    
    print('NALU RMSE learning history summarized.')
    print('\n')    
    print(nalu_rmse_history_sum)

    print('GNALU MSE learning history summarized.')
    print('\n')    
    print(gnalu_mse_history_sum)

    print('GNALU RMSE learning history summarized.')
    print('\n')        
    print(gnalu_rmse_history_sum)
