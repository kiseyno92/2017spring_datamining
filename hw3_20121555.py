
# coding: utf-8

# In[7]:

get_ipython().magic('matplotlib inline')


# In[8]:

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# test set, train set 을 불러 오고 헤더 row를 [1:]로 skip 합니다.
dataset = pd.read_csv("train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("test.csv").values


# In[10]:

# array, data type 을 수정합니다.
target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)


# In[ ]:

import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.imshow(train[1729][0], cmap=cm.binary) # draw the picture


# In[ ]:

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None,1,28,28),
        hidden_num_units=1000, # hidden layer 의 unit 개수
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 0~9 까지의 target vale

        # optimization 
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=15,
        verbose=1,
        )

# Train 
cnn = net1.fit(train, target)


# In[ ]:

# test data를 분류합니다.
pred = cnn.predict(test)

# 결과 값 저장.
np.savetxt('submission.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


# In[ ]:



