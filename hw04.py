import numpy as np
import tensorflow as tf
from tdnn import TDNN
import csv
import pandas as pd
from tensorflow.python.keras.engine import data_adapter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# def generate_dataset(n_train,filepath):
#     n_abnormal = np.random.randint(1,15)
#     T_SET = [];F_SET = []
#     for _ in range(n_train):
#         # Generate True set
#         t_idx = np.random.choice(79,20)
#         f_idx = np.random.choice(78,20)
#         t_sample = np.zeros(80)
#         f_sample = np.random.randint(2,size = 80)
#         for i in t_idx:
#             t_sample[i] = 1
#         T_SET.append(t_sample)
#         # Generate false set
#         for i in f_idx:
#             f_sample[i],f_sample[i+1] = 1,1
#         F_SET.append(f_sample)
#     T_set_filepath = filepath + "/t_set.csv"
#     f = open(T_set_filepath,'w')
#     writer = csv.writer(f)
#     for r in range(len(T_SET[:])):
#         writer.writerow(T_SET[r])
#     F_set_filepath = filepath + "/f_set.csv"
#     f = open(F_set_filepath,'w')
#     writer = csv.writer(f)
#     for r in range(len(F_SET[:])):
#         writer.writerow(F_SET[r])
#     return 
FILE_PATH = "/workspace/DeepLearning/hw04"
# generate_dataset(50,FILE_PATH)

t_dset = pd.read_csv(FILE_PATH+"/t_set.csv",skiprows = 1)
f_dset = pd.read_csv(FILE_PATH+"/f_set.csv",skiprows = 1)
t_label = np.ones((len(t_dset),1))
f_label = np.zeros((len(f_dset),1))
dset = np.vstack((t_dset,f_dset))
label = np.vstack((t_label,f_label))

test_dset = pd.read_csv(FILE_PATH+"/hmw4test.csv",skiprows = 1)
test_label = test_dset.values[:,-1]
test_data = test_dset.values[:,:80]
test_data = np.expand_dims(test_data,axis=2)
test_data = np.array(test_data,dtype='float32')
test_label = np.array(test_label,dtype='float32')
# train_loss = tf.keras.metrics.Mean()
# train_accuracy = tf.keras.metrics.BinaryAccuracy()

# test_loss = tf.keras.metrics.Mean()
# test_accuracy = tf.keras.metrics.BinaryAccuracy()

# loss_object = tf.keras.metrics.BinaryCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

def train_step(dset,label):
    with tf.GradientTape() as tape:
        predictions = tdnn(dset,training=True)
        loss = loss_object(label,predictions)
    gradients = tape.gradient(loss,tdnn.trainable_variables)
    optimizer.apply_gradients(zip(gradients,tdnn.trainable_variables))

    train_loss(loss)
    train_accuracy(label,predictions)

def test_step(dset,label):
    predictions = tdnn(dset,training=False)
    t_loss = loss_object(label,predictions)

    test_loss(t_loss)
    test_accuracy(label,predictions)

def flatten(dset,winsize):
    n_sample,l_sample = np.shape(dset)
    flatten_dset = []
    for n in range(n_sample):
        dimflat = np.zeros((l_sample-winsize,winsize))
        for i in range(l_sample-winsize):
            dimflat[i,:] = dset[n,i:i+winsize]
            # np.append(dimflat,list(dset[n,i:i+winsize]),axis=1)
            # dimflat.append(list(dset[n,i:i+winsize]))
        flatten_dset.append(dimflat)
    print(f'The shape of dataset is {np.shape(flatten_dset)}')
    return flatten_dset
    
EPOCHS = 100
X_train,X_val,y_train,y_val = train_test_split(dset,label,test_size = 0.2,random_state = 42)
winsize = 20

# X_train = flatten(X_train,winsize);X_val = flatten(X_val,winsize)
X_train = np.expand_dims(X_train,axis=2); X_val = np.expand_dims(X_val,axis=2)
tdnn = TDNN(k_size = winsize,units1 = 10, units2 = 5)
tdnn.compile(optimizer = 'Adam',
        loss = 'binary_crossentropy',
        metrics=['accuracy'])
history = tdnn.fit(X_train,y_train,epochs=500,validation_split=0.2)
tdnn.evaluate(test_data,test_label)
loss = history.history['loss']
acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
plt.subplot(121)
plt.plot(loss,'b-',label='loss')
plt.plot(acc,'r-',label='accuracy')
plt.legend()
plt.subplot(122)
plt.plot(val_loss,'b-',label='validation loss')
plt.plot(val_acc,'r-',label='validation accuracy')
plt.legend()
plt.show()
# train_ds = tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(10000).batch(1)
# val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(1)
# for epoch in range(EPOCHS):
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     test_loss.reset_states()
#     test_accuracy.reset_states()
#     i = 0
#     for d,l in train_ds:
        
#         train_step(d,l)
       
#     for d,l in test_ds:
#         test_step(d,l)

#     print(
#     f'Epoch {epoch + 1}, '
#     f'Loss: {train_loss.result()}, '
#     f'Accuracy: {train_accuracy.result() * 100}, '
#     f'Test Loss: {test_loss.result()}, '
#     f'Test Accuracy: {test_accuracy.result() * 100}'
#   )
    

