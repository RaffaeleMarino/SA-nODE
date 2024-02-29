import tensorflow as tf
import numpy as np
#import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')

def double_well_potential(z, gamma, a):
    return gamma * ((z ** 2 - a ** 2) ** 2)


def double_well_dyn(z, gamma, a):
    return 4 * gamma * (z * (a ** 2) - z ** 3)


class Linear_transfer(tf.keras.layers.Layer):
    def __init__(self,
                 rec_base,
                 units,
                 a,
                 gamma,
                 last=False
                 ):
        super(Linear_transfer, self).__init__()

        self.number_of_attractors = rec_base.shape[1]
        self.last = last
        self.units = units
        self.a=a
        self.gamma=gamma

    
        # self.base_train = tf.Variable(tf.random.normal((units, int(units - self.number_of_attractors))),
        #                               dtype=tf.float64,
        #                               trainable=True)

        # Normalization of the eigenvectors
        # self.base_train = tf.math.l2_normalize(self.base_train, axis=0)



        self.base_fix = tf.Variable(rec_base,
                                    dtype=tf.float64,
                                    trainable=False, name='base_fix')
             # Determinante per check
        self.base_train = self.add_weight(shape=(units, int(units - self.number_of_attractors)),
                                          initializer=tf.keras.initializers.Orthogonal(gain=1.15, seed=None),
                                          regularizer=tf.keras.regularizers.OrthogonalRegularizer(factor=0.01,
                                                                                                  mode='columns'),
                                          trainable=True,
                                          name='base_train')
           
        base = tf.concat([self.base_train, self.base_fix], axis=1)
        
            # n = tf.shape(base)[0]
            # detbase=tf.linalg.det(base)
            # base=tf.pow(detbase, tf.cast(-1/n, tf.float64))*base
        #tf.print('Determinant init:', tf.linalg.det(base))
        self.autov_fix = tf.Variable(tf.zeros(self.number_of_attractors, dtype=tf.float64),
                                     trainable=False, name='eigval_fix')
        self.autov_tr = tf.Variable(np.random.normal(-5, 1., int(units - self.number_of_attractors)),
                                    trainable=True,
                                    dtype=tf.float64, name='eigenval_train')
        
    def call(self, inputs, **kwargs):
        eigenvalues_total = tf.concat([tf.transpose(self.autov_tr), tf.transpose(self.autov_fix)], axis=0)

        # Create the diagonal matrix with the eigenvalues
        x_max=8.*(self.a**2)*self.gamma*np.sqrt(self.units)
        x_max=x_max-0.1
        diagonal = tf.linalg.diag(tf.clip_by_value(eigenvalues_total, clip_value_min=-1000000000., clip_value_max=x_max))

        # Concatenate the two bases
        base = tf.concat([self.base_train, self.base_fix], axis=1)
        A = tf.matmul(base, tf.matmul(diagonal, tf.linalg.inv(base)))
        return tf.matmul(inputs, tf.transpose(A))

    def return_base(self):
        return tf.concat([self.base_train, self.base_fix], axis=1)

    def return_diagonal(self):
        return tf.concat([self.autov_tr, self.autov_fix], axis=0)


class DoppiaBuca(tf.keras.Model):
    def __init__(self,
                 size,
                 tmax,
                 attractors,
                 gamma=0.2,
                 a=1.4,
                 dt=0.1
                 ):
        super(DoppiaBuca, self).__init__()

        self.size = size
        self.a = tf.Variable(a, trainable=False, dtype=tf.float64)
        self.gamma = tf.Variable(gamma, trainable=True, dtype=tf.float64)
        self.iterations = tmax
        self.attractors = attractors
        self.dt = dt
        self.beta = tf.Variable(1./np.sqrt(self.size), trainable=False) # 1.1
        self.flag=False


    def build(self, input_shape):
        self.adiacency = Linear_transfer(rec_base=self.attractors,
                                         units=self.size, a=self.a, gamma=self.gamma)

    def call(self, inputs, **kwargs):
        x = inputs
        x=tf.cast(x, dtype=tf.float64)
        for k in range(self.iterations):
            #temp_x = self.adiacency(x)
            k1=self.dt * (double_well_dyn(x, tf.abs(self.gamma), self.a) + self.adiacency(x) * self.beta)
        #    k2=self.dt * (double_well_dyn(x+0.5*k1, tf.abs(self.gamma), self.a) + self.adiacency(x+0.5*k1) * self.beta)
        #    k3=self.dt * (double_well_dyn(x+0.5*k2, tf.abs(self.gamma), self.a) + self.adiacency(x+0.5*k2) * self.beta)
        #    k4=self.dt * (double_well_dyn(x+k3, tf.abs(self.gamma), self.a) + self.adiacency(x+k3) * self.beta)
            #x = x + self.dt * (double_well_dyn(x, tf.abs(self.gamma), self.a) + temp_x * self.beta)

            x = x + k1#(k1 +2*k2 + 2*k3 + k4)/6.  #+ tf.cast(tf.math.sqrt(self.dt), dtype=tf.float64)*tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=epsilon, dtype=tf.float64)#+ (k1 +2*k2 + 2*k3 + k4)/6.
            # clip x from -2*a to 2*a
            x = tf.clip_by_value(x, -10 * self.a, 10 * self.a)
            if k % 1000 ==0:
                print(k)

            if self.flag and k%1==0:
                data=x[3,:].numpy()
                data.flatten()
                with open("output_forward_dyn.txt", "a") as f:
                    f.write(" ".join(map(str, data)) + "\n")
     
        return x
    
    def inverse_dynamics(self, inputs, **kwargs):
        x = inputs
        epsilon=0.01
        x = tf.cast(x, dtype=tf.float64)
        for k in range(self.iterations):
            k1=self.dt * (double_well_dyn(x, tf.abs(self.gamma), self.a) + self.adiacency(x) * self.beta)
            k2=self.dt * (double_well_dyn(x+0.5*k1, tf.abs(self.gamma), self.a) + self.adiacency(x+0.5*k1) * self.beta)
            k3=self.dt * (double_well_dyn(x+0.5*k2, tf.abs(self.gamma), self.a) + self.adiacency(x+0.5*k2) * self.beta)
            k4=self.dt * (double_well_dyn(x+k3, tf.abs(self.gamma), self.a) + self.adiacency(x+k3) * self.beta)
            x = x - (k1 +2*k2 + 2*k3 + k4)/6 #- tf.cast(tf.math.sqrt(self.dt), dtype=tf.float64)*tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=epsilon, dtype=tf.float64)#(k1 +2*k2 + 2*k3 + k4)/6
            x = tf.clip_by_value(x, -10 * self.a, 10 * self.a)
            if k % 1000 ==0:
                print(k)
            if self.flag and k%100==0:
                data=x[3,:].numpy()
                data.flatten()
                with open("output_back_dyn.txt", "a") as f:
                    f.write(" ".join(map(str, data)) + "\n")

            
        return x