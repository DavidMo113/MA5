#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# In[12]:


dummy_input = []
for i in range(100):
    dummy_input.append(np.random.randint(800,1000))
train_data, test_data = np.array(dummy_input[:95]), np.array(dummy_input[95:])


# In[13]:


train_data


# In[4]:


def data_batch(data):
    batch = []
    for i in range(len(data)-4):
        seq = data[i:i+4]
        label = data[i+4:i+4+1]
        batch.append((torch.tensor(seq.astype(np.float32)), torch.tensor(label.astype(np.float32))))
    return batch

batch = data_batch(train_data)
batch[:10]


# In[10]:


class Linear(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        output = self.fc(x)
        
        return output

loss = nn.MSELoss()
model = Linear(4,1)
model = model.float()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[11]:


def model_train(batch):
    epoch = 5
    class Linear(nn.Module):

        def __init__(self, input_size, output_size):
            super(Linear, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, x):
            output = self.fc(x)

            return output

    loss = nn.MSELoss()
    model = Linear(4,1)
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for i in range(epoch):
        for x,y in batch:
            optimizer.zero_grad()
            pred = model(x)
            loss_func = loss(pred, y)
            loss_func.backward()


# In[ ]:




