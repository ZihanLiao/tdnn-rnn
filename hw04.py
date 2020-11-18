import numpy as np
import tensorflow as tf
from tdnn import TDNN
import csv
import pandas as pd
from tensorflow.python.keras.engine import data_adapter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
def generate_dataset(n_train,filepath):
    n_abnormal = np.random.randint(1,15)
    T_SET = [];F_SET = []
    for _ in range(n_train):
        # Generate True set
        t_idx = np.random.choice(79,20)
        f_idx = np.random.choice(78,20)
        t_sample = np.zeros(80)
        f_sample = np.random.randint(2,size = 80)
        for i in t_idx:
            t_sample[i] = 1
        T_SET.append(t_sample)
        # Generate false set
        for i in f_idx:
            f_sample[i],f_sample[i+1] = 1,1
        F_SET.append(f_sample)
    T_set_filepath = filepath + "/t_set.csv"
    f = open(T_set_filepath,'w')
    writer = csv.writer(f)
    for r in range(len(T_SET[:])):
        writer.writerow(T_SET[r])
    F_set_filepath = filepath + "/f_set.csv"
    f = open(F_set_filepath,'w')
    writer = csv.writer(f)
    for r in range(len(F_SET[:])):
        writer.writerow(F_SET[r])
    return 
FILE_PATH = "/workspace/DeepLearning/hw04"
generate_dataset(500,FILE_PATH)

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
winsize = 7

# X_train = flatten(X_train,winsize);X_val = flatten(X_val,winsize)
X_train = np.expand_dims(X_train,axis=2); X_val = np.expand_dims(X_val,axis=2)
tdnn = TDNN(k_size = winsize,units1 = 20, units2 = 20)
tdnn.compile(optimizer = 'Adam',

        loss = 'binary_crossentropy',
        metrics=['accuracy'])
history = tdnn.fit(X_train,y_train,epochs=10,validation_split=0.2)
predictions = tdnn.predict(test_data)
predictions = np.where(predictions>0.5,1,0)
tdnn.evaluate(test_data,test_label)
cm = confusion_matrix(test_label,predictions)
loss = history.history['loss']
acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
plt.figure(1)
plt.subplot(121)
plt.plot(loss,'b-',label='loss')
plt.plot(acc,'r-',label='accuracy')
plt.legend()
plt.subplot(122)
plt.plot(val_loss,'b-',label='validation loss')
plt.plot(val_acc,'r-',label='validation accuracy')
plt.legend()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds): 
    if normalize: 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        print("Normalized confusion matrix") 
    else: 
        print('Confusion matrix, without normalization') 
        # print(cm) 
    plt.imshow(cm, interpolation='nearest', cmap = 'Blues') 
    plt.title(title)
    tick_marks = np.arange(len(classes)) 
    plt.xticks(tick_marks, classes, rotation=45) 
    plt.yticks(tick_marks, classes) 
    fmt = '.2f' if normalize else 'd' 
    thresh = cm.max() / 2. 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, format(cm[i, j], fmt), 
            horizontalalignment="center", 
            color="white" if cm[i, j] > thresh else "black") 
    plt.tight_layout() 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    plt.show()
plt.figure(2)
plot_confusion_matrix(cm,[0,1],normalize=True,title='confusion matrix(tdnn)')

    

