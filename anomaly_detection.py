# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:02:48 2020

Benchmark of [unsupervided] anomaly detection algorithms based on Machine Learning
    K-means
    Isolation forrest
    Autoencoder 
        
Based on dataset found here : https://github.com/numenta/NAB/blob/master/data/realKnownCause/machine_temperature_system_failure.csv
Related paper : https://www.sciencedirect.com/science/article/pii/S0925231217309864

Documentation/ Reference :
K means : http://amid.fish/anomaly-detection-with-k-means-clustering
Autoencoder : http://philipperemy.github.io/anomaly-detection/
Isolation forest : scikitlearn
    
Personal notes :
Intuition behind K-means and Autoencoder : construct time-basis functions with a reduced dimension and project new sequences on it
Further Ideas : Projection of sequences on wavelet basis. 
In order to improve AE, applying GAN

@author: Thibaut Le Magueresse
"""
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras import layers
from datetime import datetime

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
#%% Raw data
df = pd.read_csv('machine_temperature_system_failure.csv', parse_dates=['timestamp'], index_col='timestamp')
stopping_training_date = '2013-12-15 00:00:00'
plt.figure()
plt.subplot(221)
plt.plot(df,label='Full recording')
plt.vlines(df.index[df.index==stopping_training_date][0], min(df.value), max(df.value), linestyles='dashed',label='Stopping training date')
plt.xlabel('Date time')
plt.ylabel('Temperature [°F]')
plt.legend()
plt.title('Time Series')

plt.subplot(222)
plt.hist(df.value, bins = 50, alpha=0.5)
plt.xlabel('Temperature values')
plt.ylabel('Frequency of appearence')

print('Total duration of signal : ', df.index[-1] - df.index[0])
print('Training duration : ', df.index[df.index  == stopping_training_date][0] - df.index[0])
# %% Pre-processing the data
train, test = df.iloc[df.index <= stopping_training_date], df.iloc[df.index > stopping_training_date]
test_size = len(train)
print(train.shape, test.shape)

scaler = StandardScaler().fit(train[['value']])
train_scaled = scaler.transform(train[['value']])
test_scaled = scaler.transform(test[['value']])

def create_dataset(X, slide_len=1, segment_len = 100, windowing = True):
    # Function that create sequences for the Recurrent Network 
    # Overlap of "slide_len" sample
    segments = []
    for start_pos in range(0, len(X), slide_len):
        end_pos = start_pos + segment_len
        # make a copy so changes to 'segments' doesn't modify the original X
        segment = np.copy(X[start_pos:end_pos])
        # if we're at the end and we've got a truncated segment, drop it
        if len(segment) != segment_len:
            continue
        if(windowing): # Very important to avoid side effect
            segments.append(np.hanning(segment_len)[:,np.newaxis] * segment)
        else:
            segments.append(segment)
    return np.squeeze(np.array(segments))

segment_len = 200 # Lenght of the sequence
X_train = create_dataset(train_scaled,segment_len = segment_len)

plt.figure()
for i in range(12): plt.subplot(3,4,i+1),plt.plot(X_train[int(X_train.shape[0]/12)*i,:])
plt.suptitle('Exemples of sequences from training dataset')

# %% K means
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=20)
clusterer.fit(X_train)
centroids = clusterer.cluster_centers_
plt.figure()
for i in range(12): plt.subplot(3,4,i+1),plt.plot(centroids[i,:])
plt.suptitle('Exemple of basis function from KMeans algorithm')


# %% Test it on the whole dataset
slide_len = int(segment_len / 2) # Overlapp 50%
X_test = create_dataset(test_scaled,slide_len=slide_len ,segment_len = segment_len)


reconstruction = np.zeros(len(test_scaled))
for segment_n in range(X_test.shape[0]):
    seg_tested = np.copy(X_test[segment_n,:])
    nearest_centroid_idx = clusterer.predict(seg_tested[np.newaxis,:])[0]
    nearest_centroid = np.copy(centroids[nearest_centroid_idx])
    
    # overlay our reconstructed segments with an overlap of half a segment
    pos = int(segment_n * X_test.shape[1]/2)
    if reconstruction[segment_n*slide_len:segment_n*slide_len+segment_len].shape[0] != segment_len:
        continue
    reconstruction[segment_n*slide_len:segment_n*slide_len+segment_len] += nearest_centroid


error = abs(reconstruction - np.squeeze(test_scaled))
threshold = 1.5
indx_anomaly = np.where(error>threshold)[0]

fig, ax1 = plt.subplots()

ax1.set_xlabel('index sample')
ax1.set_ylabel('Scaled Temperature')
ax1.plot(test_scaled,  label="True data")
ax1.plot(reconstruction,  label="econstructed data")

ax1.scatter(indx_anomaly,reconstruction[indx_anomaly],color='black',label = 'anomalies')
ax1.legend()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:gray'
ax2.set_ylabel('Error', color=color)  # we already handled the x-label with ax1
ax2.plot(error, color=color,linestyle='dashed' , alpha=0.5)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# %% Isolation forrest
#specify the 12 metrics column names to be modelled
from sklearn.ensemble import IsolationForest
clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.002), \
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf.fit(train_scaled)
pred = clf.predict(test_scaled)
indx_anomaly_isolation=np.where(pred==-1)
plt.figure()
plt.plot(test_scaled)
plt.scatter(indx_anomaly_isolation,test_scaled[indx_anomaly_isolation],color='black',label = 'anomalies')


# %% Test d'idées : autoencoder based on wavelet decomposition of sequences
# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)

def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()

nb_epochs = 25
batch_size = 64
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history

# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()


# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])

reconstruction_ae = np.zeros(len(test_scaled))

for segment_n in range(X_test.shape[0]):
    seg_tested = np.copy(X_test[segment_n,:])
    # overlay our reconstructed segments with an overlap of half a segment
    pos = int(segment_n * X_test.shape[1]/2)
    if reconstruction_ae[segment_n*slide_len:segment_n*slide_len+segment_len].shape[0] != segment_len:
        continue
    reconstruction_ae[segment_n*slide_len:segment_n*slide_len+segment_len] += X_pred[segment_n,:]

error = abs(reconstruction_ae - np.squeeze(test_scaled))
threshold = 1.5
indx_anomaly_ae = np.where(error>threshold)[0]

fig, ax1 = plt.subplots()

ax1.set_xlabel('index sample')
ax1.set_ylabel('Scaled Temperature')
ax1.plot(test_scaled,  label="True data")
ax1.plot(reconstruction_ae,  label="econstructed data")

ax1.scatter(indx_anomaly_ae,reconstruction_ae[indx_anomaly_ae],color='black',label = 'anomalies')
ax1.legend()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:gray'
ax2.set_ylabel('Error', color=color)  # we already handled the x-label with ax1
ax2.plot(error, color=color,linestyle='dashed' , alpha=0.5)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%% Displays : Synthesis
indx_ground_truth = np.array([495,12853,16025])
plt.figure()
plt.subplot(221)
plt.plot(test_scaled,  label="True data")
plt.scatter(indx_ground_truth,test_scaled[indx_ground_truth],color='black',label = 'anomalies')
plt.title('Ground Truth')
ax1.set_xlabel('index sample')
ax1.set_ylabel('Scaled Temperature')

plt.subplot(222)
plt.plot(test_scaled,  label="True data")
plt.plot(reconstruction_ae,  label="Reconstructed by AE")
plt.scatter(indx_anomaly_ae,test_scaled[indx_anomaly_ae],color='black',label = 'anomalies')
plt.title('AutoEncoder - LSTM')
ax1.set_xlabel('index sample')
ax1.set_ylabel('Scaled Temperature')

plt.subplot(223)
plt.plot(test_scaled,  label="True data")
plt.scatter(indx_anomaly_isolation,test_scaled[indx_anomaly_isolation],color='black',label = 'anomalies')
plt.title('Isolation Forrest')
ax1.set_xlabel('index sample')
ax1.set_ylabel('Scaled Temperature')

plt.subplot(224)
plt.plot(test_scaled,  label="True data")
plt.plot(reconstruction,  label="Reconstructed by K-means")
plt.scatter(indx_anomaly,test_scaled[indx_anomaly],color='black',label = 'anomalies')
plt.title('K-means')
ax1.set_xlabel('index sample')
ax1.set_ylabel('Scaled Temperature')