import os
import tensorflow as tf
import pickle
from functions import *

suffix = 'MOE' # 'PINN' OR 'NPI'
  
train_stop = 2015

time_vars, space_vars, depth, max_depth, temperature, year = load_data()
datasets, _ = split_data(time_vars, space_vars, depth, max_depth, temperature, year, train_stop)

train_dataset = tf.data.Dataset.from_tensor_slices((datasets['X_train'], datasets['y_train']))
train_dataset = train_dataset.shuffle(buffer_size=len(datasets['y_train'])).batch(256)

val_dataset = tf.data.Dataset.from_tensor_slices((datasets['X_val'], datasets['y_val']))
val_dataset = val_dataset.batch(256)

with tf.device('/GPU:0'):

  shape_t = datasets['X_train'][0].shape[1:] 
  shape_s = datasets['X_train'][1].shape[1:]
  
  sizes_t = [384,256,128]
  sizes_s = [256,128]
  sizes_c = [256,128]
  
  drop_t = [0.2,0.2,0.2]
  drop_s = [0., 0.]
  drop_c =[0.2,0.2]
  
  model = build_st_model(shape_t, shape_s, sizes_t, sizes_s, sizes_c, drop_t, drop_s, drop_c, suffix)
  optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-4)
  model.compile(optimizer=optimizer, loss='mse')

log_filename = os.path.join(os.getcwd(), f"history_{suffix}.txt")
loss_logger = LossLogger(log_filename)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=20, factor=0.8, min_lr=1e-6)

callbacks = [loss_logger, reduce_lr]

model.fit(train_dataset, validation_data=val_dataset, epochs=1000, verbose=0, callbacks=callbacks);
model.save(f'./models/model_{suffix}.keras')



















