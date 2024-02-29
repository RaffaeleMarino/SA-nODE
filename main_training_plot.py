import os
from os.path import join
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

#use_pretrained = True #no training
use_pretrained = False
noise=0.0 #mi fermo a 0.45 perchè osservo cose strana sulla stabilità

# Load the dataset
DM = DatasetMaker(config)
#qui devo partire con la creazione del mio dataset
if use_pretrained:
    (flat_train, y_train), (flat_test, y_test) = DM.load_MNIST_dataset(perturb=noise)
    attractors = DM.attractor_matrix
    model_path='final_model0.0.tf'
    model = tf.keras.models.load_model(model_path)
else:
    (flat_train, y_train), (flat_test, y_test) = DM.load_MNIST_dataset(perturb=noise)
    attractors = DM.attractor_matrix


y_train=y_train[:, :10] @ attractors.T
y_test=y_test[:, :10] @ attractors.T

# Orthogonality of attractors (Space preserved)
attractors = np.linalg.qr(attractors)[0]
t_max = 55

# %%
# Model
if not use_pretrained:
    model = DoppiaBuca(size=config['Network']['size'],
                       attractors=attractors,
                       tmax=t_max,
                       **config['DynamicalParameters'])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    model(flat_train)
   # model.evaluate(x=flat_train,y=y_train,verbose=1)
    indice_strato = None
    for i, layer in enumerate(model.layers):
        if 'Linear_transfer' in str(type(layer)):
         indice_strato = i
         break
    print(indice_strato)
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
    plt.hist(eigenvalues_total.flatten(), bins=784)
    plt.xlabel('Autovalori init')
    plt.ylabel('Frequenza')
    plt.savefig('histo_init.jpg')
    plt.show()
    plt.close()
    x1=8*(model.a.numpy()**2)*model.gamma.numpy()*np.sqrt(784)
    values_between = eigenvalues_total[(eigenvalues_total >= x1)]
    print(values_between)
    model.save('initial_model'+str(noise)+'.tf')
    model.fit(x=flat_train,
              y=y_train,
              shuffle=True,
              batch_size=1000,
              epochs=20,
              verbose=1)
    model.save('final_model'+str(noise)+'.tf')
    indice_strato = None
    for i, layer in enumerate(model.layers):
        if 'Linear_transfer' in str(type(layer)):
         indice_strato = i
         break
    print(indice_strato)
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
    plt.hist(eigenvalues_total.flatten(), bins=784)
    plt.xlabel('Autovalori final')
    plt.ylabel('Frequenza')
    plt.savefig('histofinal.jpg')
    plt.show()
    plt.close()
    x1=8*(model.a.numpy()**2)*model.gamma.numpy()*np.sqrt(784)
    values_between = eigenvalues_total[(eigenvalues_total >= x1)]
    print(values_between)
    

else:
    modeln = DoppiaBuca(size=config['Network']['size'],
                       attractors=attractors,
                       tmax=t_max,
                       **config['DynamicalParameters'])

    modeln.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')

    modeln(flat_train)
    modeln.set_weights(model.get_weights())
    modeln(flat_train)
    print('Model loaded from directory: {}'.format(model_path))
    modeln.summary()
    modeln.fit(x=flat_train,
              y=y_train,
              shuffle=True,
              batch_size=500,
              epochs=200,
              verbose=1)
    modeln.save('final_model'+str(noise)+'.tf')
    indice_strato = None
    for i, layer in enumerate(modeln.layers):
        if 'Linear_transfer' in str(type(layer)):
         indice_strato = i
         break
    print(indice_strato)
    weights = modeln.layers[indice_strato].get_weights()
    layer = modeln.layers[indice_strato]
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
    plt.hist(eigenvalues_total.flatten(), bins=784)
    plt.xlabel('Autovalori final')
    plt.ylabel('Frequenza')
    plt.savefig('histofinal.jpg')
    plt.show()
    plt.close()
    x1=8*(model.a.numpy()**2)*model.gamma.numpy()*np.sqrt(784)
    values_between = eigenvalues_total[(eigenvalues_total >= x1)]
    print(values_between)

y_pred=modeln(flat_test)
counter=0
y_pred=np.round(y_pred.numpy() * 2) 
y_pred=np.sign(y_pred)/2.
for i in range(np.shape(y_test)[0]):
    if  np.sum(np.abs(y_test[i,:]-y_pred[i,:]))==0:
        counter=counter+1.
print('accuracy')
print(counter/np.shape(y_test)[0])
