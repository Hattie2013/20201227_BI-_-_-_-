#!/usr/bin/env python
# coding: utf-8

# In[7]:


#步骤一 数据准备：将原始数据格式中的year, month, day, hour进行合并，并保存新的文件pollution.csv
import pandas as pd
from datetime import datetime

#步骤二 加载数据并完成数据规范化，包括：1）删除非特征列‘No’，2）替换列名， 3）线性插补缺漏值， 4）删除前24小时的数据
def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')
dataset = pd.read_csv('./raw.csv', parse_dates = [['year','month','day','hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
dataset.columns = ['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
dataset.index.name = 'date'
dataset['pollution'].fillna(0, inplace=True)
dataset = dataset[24:]
dataset.to_csv('pollution.csv')
dataset


# In[8]:


import pandas as pd

dataset = pd.read_csv('./pollution.csv', index_col=0)
values = dataset.values
values


# In[9]:


import matplotlib.pyplot as plt
import pandas as pd

#步骤三 确认可视化特征共8个
dataset = pd.read_csv('./pollution.csv', index_col=0)
values = dataset.values

i = 1
for group in range(8):
    plt.subplot(8, 1, i)
    plt.plot(values[:, group])
    plt.title(dataset.columns[group])
    i = i + 1
plt.show()


# In[10]:


dataset['wnd_dir'].value_counts()


# In[11]:


#步骤四 将分类特征wnd_dir进行标签编码
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype('float32')
values


# In[12]:


values[:, 4]


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)
scaled.shape


# In[16]:


#步骤五 将时序数据转换为监督学习数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]   
    for i in range (0, n_out):
        cols.append(df.shift(-i))
        if i == 0 :
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg        


# In[17]:


#步骤六 将时间序列数据转换为适合监督学习的数据
reframed = series_to_supervised(scaled, 1, 1) 
reframed.to_csv('reframed-1.csv')
reframed


# In[18]:


#步骤七 去掉不需要预测的列
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
reframed.to_csv('reframed-2.csv')
reframed


# In[19]:


#步骤八 数据集切分并启动训练
values = reframed.values
n_train_hours = int(len(values) * 0.8)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X.shape


# In[20]:


train_y.shape


# In[21]:


#步骤九 转换为3D格式
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#步骤十 设置网络模型并进行模型训练
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
result = model.fit(train_X, train_y, epochs=10, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[25]:


#步骤十一 模型预测
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)
test_predict


# In[29]:


#步骤十二 绘制训练损失和测试损失
line1 = result.history['loss']
line2 = result.history['val_loss']
plt.plot(line1, label='train', c='g')
plt.plot(line2, label='test', c='r')
plt.legend(loc='best')
plt.show()


# In[30]:


model.summary()


# In[35]:


#步骤十三 原始数据，训练结果，测试结果
def plot_img(source_data_set, tranin_predict, test_predict):
    plt.plot(source_data_set[:, -1], label='real', c='b')
    plt.plot([x for x in train_predict], label='train_predict', c='g')
    temp = [None for _ in train_predict] + [x for x in test_predict]
    plt.plot(temp, label='test_predict', c='r')
    plt.legend(loc='best')
    plt.show()

plot_img(values, train_predict, test_predict)

