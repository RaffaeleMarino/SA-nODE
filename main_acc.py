import os
import sys
#definisco le istruzioni per la GPU

dev = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = dev
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')  
    tf.config.experimental.set_memory_growth(physical_devices[0], True)    
    print("GPU available.")
else:
    print("No GPU available. Using CPU instead.")


#Fine istruzioni  per la GPU
import matplotlib
import matplotlib.pyplot as plt
from functions import DatasetMaker, LoadConfig
from models import DoppiaBuca
import numpy as np
import shutil

# Set up the config and log dir
config = LoadConfig('doppiabuca.yml')
log_dir = ".\\logs"
# clear log dir
shutil.rmtree(log_dir,
              ignore_errors=True)
# check if the directory is deleted correctly
if os.path.exists(log_dir):
    raise Exception('Directory not deleted')

use_pretrained = True #no training
#use_pretrained = False #si training
noise=0.0
# Load the dataset
DM = DatasetMaker(config)
if use_pretrained:
    model_path='./MNISTandFASHION/final_model0.0-TANH-MNIST.tf'
else:
    model_path='initial_model.tf'
model = tf.keras.models.load_model(model_path)


(flat_train, y_train), (flat_test, y_test) = DM.load_MNIST_dataset(perturb=noise)
attractors = DM.attractor_matrix



y_train=y_train[:, :11] @ attractors.T
y_test=y_test[:, :11] @ attractors.T

# Orthogonality of attractors (Space preserved)
attractors = np.linalg.qr(attractors)[0]
t_max = 600



#analisi della distribuzione degli autovalori
indice_strato = None
for i, layer in enumerate(model.layers):
    if 'Linear_transfer' in str(type(layer)):
        indice_strato = i
        break

weights = model.layers[indice_strato].get_weights()
layer = model.layers[indice_strato]
specific_weight = None
for weight in layer.weights:
    if 'eigenval_train' in weight.name:
        specific_weight = weight.numpy()
        break

specific_weight1 = None
for weight in layer.weights:
    if 'eigval_fix' in weight.name:
        specific_weight1 = weight.numpy()
        break

eigenvalues_total = np.concatenate((specific_weight, specific_weight1), axis=0)


if not use_pretrained:
    model = DoppiaBuca(size=config['Network']['size'],
                       attractors=attractors,
                       tmax=t_max,
                       **config['DynamicalParameters'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
            
    model.evaluate(x=flat_train,y=y_train,verbose=1)
    model.fit(x=flat_train,
              y=y_train,
              shuffle=True,
              batch_size=200,
              epochs=700,
              verbose=1)
else:
    model(flat_test[0:2])
    print('Model loaded from directory: {}'.format(model_path))
    #model.summary()

# %%
# Create a new Double Well model and load the weights of the trained model
new_model = DoppiaBuca(size=config['Network']['size'],
                       attractors=attractors,
                       tmax=1200,
                       **config['DynamicalParameters'])

new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
#new_model1 = DoppiaBuca(size=config['Network']['size'],
#                       attractors=attractors,
#                       tmax=t_max,
#                       **config['DynamicalParameters'])

#new_model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                  loss='mse')

new_model(flat_train[0:2])
new_model.set_weights(model.get_weights())

# %%
y_pred=new_model(flat_test)
#mat_shape = (28, 28)
#input_vec = flat_test[0,:]
#fig, ax = plt.subplots(3, 1, figsize=(28, 28))
#ax[0].imshow(input_vec.reshape(mat_shape),cmap='gray')
#ax[0].set_title('Input', fontsize=60)
#out_iteration = y_pred[0,:].numpy()
#ax[1].imshow(out_iteration.reshape(mat_shape),cmap='gray')
#ax[1].set_title('Output t = {}'.format(new_model.iterations*0.1), fontsize=60)
#ax[2].imshow(y_test[0, :].reshape(mat_shape),cmap='gray')
#ax[2].set_title('Target', fontsize=60)
#plt.tight_layout()
#plt.savefig('fig_ok'+str(0)+'.jpg')
#plt.close()
#sys.exit(-1)
y_pred2=y_pred
res1=np.trace(((y_pred/model.a.numpy()) @ (y_test.T/model.a.numpy()))/784)/np.shape(y_test)[0]
counter=0
counter_less=0
#y_pred1=new_model1(flat_test) 
y_pred2=np.sign(y_pred2)/2.
for i in range(np.shape(y_test)[0]):
    if  np.sum(np.abs(y_test[i,:]-y_pred2[i,:]))==0.:
        counter=counter+1.
    elif np.sum(np.abs(y_test[i,:]-y_pred2[i,:]))<100:
         if (np.sign(y_test[i,:]) @ np.sign(y_pred2[i,:]))/784 > 0.95:
            counter_less=counter_less+1
         mat_shape = (28, 28)
         input_vec = flat_test[i,:]
         fig, ax = plt.subplots(3, 1, figsize=(28, 28))
         ax[0].imshow(input_vec.reshape(mat_shape),cmap='gray')
         ax[0].set_title('Input', fontsize=60)
         out_iteration = y_pred[i,:].numpy()
         ax[1].imshow(out_iteration.reshape(mat_shape),cmap='gray')
         ax[1].set_title('Output t = {}'.format(new_model.iterations*0.1), fontsize=60)
         ax[2].imshow(y_test[i, :].reshape(mat_shape),cmap='gray')
         ax[2].set_title('Target', fontsize=60)
         plt.tight_layout()
         plt.savefig('fig_error'+str(i)+'.jpg')
         plt.close()
         print('error fig_error'+str(i)+'.jpg :', np.sum(np.abs(y_test[i,:]-y_pred2[i,:])), (np.sign(y_test[i,:]) @ np.sign(y_pred2[i,:]))/784)
print('accuracy')
plt.imshow(flat_test[0,:].reshape(28,28))
plt.savefig('fig_prova.jpg')
print(model_path, noise, counter/np.shape(y_test)[0], counter_less, (counter+counter_less)/np.shape(y_test)[0], res1)
sys.exit(-1)
# Plotting
for i in range(20):
    n = np.random.randint(0, len(flat_test))
    mat_shape = (28, 28)
    input_vec = flat_test[n:n + 1, :]

    fig, ax = plt.subplots(4, 1, figsize=(28, 28))

# Input vector
    ax[0].imshow(input_vec.reshape(mat_shape))
    ax[0].set_title('Input')

# Output of the trained model
    out = new_model1(input_vec).numpy()
    ax[1].imshow(out.reshape(mat_shape))
    ax[1].set_title('Output t = {}'.format(t_max))

# Output of the new model with augmented iterations
    out_iteration = new_model(input_vec).numpy()
    ax[2].imshow(out_iteration.reshape(mat_shape))
    ax[2].set_title('Output t = {}'.format(new_model.iterations))

# Target
    ax[3].imshow(y_test[n:n + 1, :].reshape(mat_shape))
    ax[3].set_title('Target')


    plt.tight_layout()
    plt.savefig('fig'+str(i)+'.jpg')
    plt.show()
    plt.close()

