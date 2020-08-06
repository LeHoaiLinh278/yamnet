# %%
import os
import pandas as pd
import numpy as np
import tensorflow
from utils import read_wav
import params
import resampy
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import features as features_lib
import tensorflow as tf
import ast 

# %%
def class_names(class_map_csv):
    """Read the class name definition file and return a list of strings."""
    with open(class_map_csv) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)   # Skip header
        return np.array([display_name for (_, _, display_name) in reader])
# %%
class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, data_frame, batch_size=1, shuffle=True, augment=False):
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = None
        # encoder labels
        self.encoder_labels = self.encoder_label()
        #self.classes_name = self.encoder_label().classes_
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_frame) / self.batch_size))
        #return int(len(self.data_frame))

    def encoder_label(self):
        n_classes = class_names('yamnet_class_map.csv')
        #print(MultiLabelBinarizer()
        return MultiLabelBinarizer(classes=n_classes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        labels = []
        #print('index', index)
        #print(len(indexes))
        self.subsequent_class = False
        for i in indexes:
            row_data = self.data_frame.iloc[i]
            waveform, sr = read_wav(row_data['filename'])
            if self.augment == False:
                # if len(waveform.shape) > 1:
                #     waveform = np.mean(waveform, axis=1)
                # if sr != params.SAMPLE_RATE:
                #     waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

                waveform = np.reshape(waveform, [1, -1])
                spectrogram = features_lib.waveform_to_log_mel_spectrogram(
                            tf.squeeze(waveform, axis=0), params)
                patches = features_lib.spectrogram_to_patches(spectrogram, params)
                for i in range(patches.shape[0]):
                    labels.append(row_data['labels'])
                if self.subsequent_class == True:
                    features = np.concatenate((features, patches))
                else:
                    features = patches
                    self.subsequent_class = True 
        labels = [ast.literal_eval(i) for i in labels]
        labels = self.encoder_labels.fit_transform(labels)
        #print(np.shape(features), labels.shape)
        return features, labels
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_frame))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

# %%
class DataGenerator_Feature(tensorflow.keras.utils.Sequence):
    def __init__(self, data_frame, batch_size=1, shuffle=True, augment=False, path_dir= 'balanced_feature'):
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = None
        self.path_dir = path_dir
        # encoder labels
        self.encoder_labels = self.encoder_label()
        #self.classes_name = self.encoder_label().classes_
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_frame) / self.batch_size))
        #return int(len(self.data_frame))

    def encoder_label(self):
        n_classes = class_names('yamnet_class_map.csv')
        #print(MultiLabelBinarizer()
        return MultiLabelBinarizer(classes=n_classes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print('index', index)
        #print(len(indexes))
        self.subsequent_class = False
        for i in indexes:
            row_data = self.data_frame.iloc[i]
            data_df = pd.read_hdf(self.path_dir + '/' + row_data['filename'].split('/')[-1].split('.')[0] + '.h5')
            read_input = data_df.values
            if self.augment == False:
                if self.subsequent_class == True:
                    features = np.concatenate((features, read_input[0][1]))
                    labels = np.concatenate((labels, read_input[0][2]))
                else:
                    features = read_input[0][1]
                    labels = read_input[0][2]
                    self.subsequent_class = True 
        #print(np.shape(features), labels.shape)
        return features, labels

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_frame))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


# %%
# df_data = pd.read_csv('unbalanced_file_train.csv')
# features, labels = DataGenerator(df_data).__getitem__(0)
# n_classes = DataGenerator(df_data).encoder_label()
# print(len(labels[0]))

# # %%
# print(labels[0])
# print(np.where(labels[0] == 1))
# %%
