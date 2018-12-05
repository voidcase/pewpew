# attempt 1
## setup
128x128
color
timestamps matched via h5-master timestamp
## result
100/100 [==============================] - 4s 44ms/step - loss: 98640.4863 - mean_squared_error: 98640.4863
100/100 [==============================] - 4s 36ms/step - loss: 13696.4021 - mean_squared_error: 13696.4021
100/100 [==============================] - 4s 36ms/step - loss: 747.8550 - mean_squared_error: 747.8550
100/100 [==============================] - 4s 36ms/step - loss: 12.7700 - mean_squared_error: 12.7700
100/100 [==============================] - 4s 36ms/step - loss: 887.2500 - mean_squared_error: 887.2500
100/100 [==============================] - 4s 37ms/step - loss: 6666.3800 - mean_squared_error: 6666.3800
100/100 [==============================] - 4s 37ms/step - loss: 77860.8000 - mean_squared_error: 77860.8000
## notes
loss probably rises due to new sample

# attempt 2
## setup
128x128
color
timestamps matched via h5-master timestamp
## result
500/500 [==============================] - 18s 36ms/step - loss: 18848.7985 - mean_squared_error: 18848.7985
500/500 [==============================] - 17s 35ms/step - loss: 11020.2399 - mean_squared_error: 11020.2399
500/500 [==============================] - 17s 35ms/step - loss: 38080.1850 - mean_squared_error: 38080.1850
500/500 [==============================] - 17s 35ms/step - loss: 27382.8472 - mean_squared_error: 27382.8472

# attempt 3
## setup
128x128
color
timestamps matched via h5-master timestamp
now with validation set
## result
Train on 100 samples, validate on 168 samples
Epoch 1/1
100/100 [==============================] - 6s 59ms/step - loss: 106086.9900 - val_loss: 1415.4464
Train on 100 samples, validate on 168 samples
Epoch 1/1
100/100 [==============================] - 5s 55ms/step - loss: 1476.8300 - val_loss: 1415.4464
Train on 100 samples, validate on 168 samples
Epoch 1/1
100/100 [==============================] - 5s 54ms/step - loss: 37.9800 - val_loss: 1415.4464
## notes
confirmation that this is overfitting like hell, gonna shuffle the training set

# attempt 4
## setup
128x128
color
timestamps matched via h5-master timestamp
shuffled training data
## result
300/300 [==============================] - 13s 43ms/step - loss: 42982.5668 - val_loss: 2291.0052
Train on 300 samples, validate on 168 samples
Epoch 1/1
300/300 [==============================] - 12s 41ms/step - loss: 25293.5485 - val_loss: 852.7251
Train on 300 samples, validate on 168 samples
Epoch 1/1
300/300 [==============================] - 12s 41ms/step - loss: 35246.5171 - val_loss: 814.9686
Train on 300 samples, validate on 168 samples
Epoch 1/1
300/300 [==============================] - 13s 42ms/step - loss: 33950.7062 - val_loss: 1414.7905
Train on 300 samples, validate on 168 samples
Epoch 1/1
300/300 [==============================] - 13s 42ms/step - loss: 14045.5806 - val_loss: 1414.7136
