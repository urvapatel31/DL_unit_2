#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np



# In[2]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape


# In[3]:


X_test.shape


# In[4]:


y_train.shape


# In[5]:


y_train


# In[6]:


y_train = y_train.reshape(-1,)
y_train


# In[7]:


y_test = y_test.reshape(-1,)


# In[8]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[9]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[10]:


plot_sample(X_train, y_train, 0)


# In[11]:


plot_sample(X_train, y_train, 1)


# In[12]:


# Normalizing the training data

X_train = X_train / 255.0
X_test = X_test / 255.0


# In[13]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


# In[14]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))



# In[15]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[16]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[17]:


cnn.fit(X_train, y_train, epochs=10)



# In[18]:


y_pred = cnn.predict(X_test)
y_pred


# In[19]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes


# In[20]:


y_test


# In[21]:


plot_sample(X_test, y_test,3)


# In[23]:


classes[y_classes[3]]

