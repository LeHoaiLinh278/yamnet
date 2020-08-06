# %%
import sys
import os 

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import pandas as pd
#from sklearn.model_selection import train_test_split
from data_generator import DataGenerator, DataGenerator_Feature
import yamnet_mobilenetv1
import params
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, applications
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from utils import read_wav
import resampy
import features as features_lib
import numpy as np
#import yamnet_mobilenetv2
# %%
# waveform, sr = read_wav('unbalanced_train_segments/Y--_5esUcUAk.wav')
# if len(waveform.shape) > 1:
#     waveform = np.mean(waveform, axis=1)
# if sr != params.SAMPLE_RATE:
#     waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

# waveform = np.reshape(waveform, [1, -1])
# spectrogram = features_lib.waveform_to_log_mel_spectrogram(
#     tf.squeeze(waveform, axis=0), params)

# print(spectrogram.shape)
# patches = features_lib.spectrogram_to_patches(spectrogram, params)
# print(patches.shape)

# %%
# configuration
BATCH_SIZE = 8
# %%

train_df = pd.read_csv('balanced_data_train.csv')
val_df = pd.read_csv('eval_data_train.csv')

# train_df, val_df = train_test_split(df_data, test_size=0.2, random_state=42)
# train_df = train_df.reset_index()
# val_df = val_df.reset_index()

# train_gen = DataGenerator(train_df, batch_size=BATCH_SIZE, shuffle=True, augment=False)
# val_gen = DataGenerator(val_df, batch_size=BATCH_SIZE, shuffle=True, augment=False)

train_gen = DataGenerator_Feature(train_df, path_dir='balanced_feature' ,batch_size=BATCH_SIZE, shuffle=True, augment=False)
val_gen = DataGenerator_Feature(val_df, path_dir='eval_feature' ,batch_size=BATCH_SIZE, shuffle=False, augment=False)

#%%

model = yamnet_mobilenetv1.yamnet()
#model = yamnet_mobilenetv2.yamnet()

model.summary()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], optimizer=optimizers.Adam(lr=0.001)) # metrics=['accuracy'] loss=[categorical_focal_loss(alpha=.25, gamma=2)] loss='categorical_crossentropy'

# %%
MAXEPOCHS = 200
LEARNING_RATE = 0.001
LEARNING_RATE_SCHEDULE_FACTOR = 0.05
LEARNING_RATE_SCHEDULE_PATIENCE = 5
mcp = ModelCheckpoint("mobilenetv1_audioset.h5", monitor="val_accuracy", save_best_only=True, save_weights_only=True, verbose=1, mode='max') #monitor="val_accuracy"
rlr = ReduceLROnPlateau(monitor='val_accuracy', factor=LEARNING_RATE_SCHEDULE_FACTOR, mode='max', patience=LEARNING_RATE_SCHEDULE_PATIENCE, min_lr=1e-10, verbose=1) #min_lr=1e-8
# earlystopping ends training when the validation loss stops improving
earlystop = EarlyStopping(monitor='val_accuracy', patience=15, verbose=0, mode='max')


steps_per_epoch =  train_gen.__len__() // BATCH_SIZE
print('steps_per_epoch', steps_per_epoch, train_gen.__len__())
validation_steps = val_gen.__len__() // BATCH_SIZE
print('steps_per_epoch', validation_steps, val_gen.__len__())
# history = model.fit_generator(train_gen,
#                                 #steps_per_epoch=steps_per_epoch,
#                                 epochs=100,
#                                 verbose=1,
#                                 validation_data=val_gen,
#                                 #validation_steps=validation_steps,
#                                 callbacks=[mcp, rlr, earlystop])

history = model.fit_generator(train_gen,
                    steps_per_epoch=train_gen.__len__(),
                    epochs=MAXEPOCHS,
                    verbose=1,
                    validation_data=val_gen,
                    validation_steps=val_gen.__len__(),
                    use_multiprocessing=False,
                    callbacks=[mcp, rlr, earlystop])

# %%
