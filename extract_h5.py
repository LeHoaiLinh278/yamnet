# %%
import os
import numpy as np
import h5py
import soundfile
import glob
import pandas as pd
import params
import ast 
import csv
from sklearn.preprocessing import MultiLabelBinarizer
import resampy
from utils import read_wav
import features as features_lib
import tensorflow as tf
import tables
from tqdm import tqdm
# %%
def class_names(class_map_csv):
    """Read the class name definition file and return a list of strings."""
    with open(class_map_csv) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)   # Skip header
        return np.array([display_name for (_, _, display_name) in reader])

def encoder_label():
    n_classes = class_names('yamnet_class_map.csv')
    return MultiLabelBinarizer(classes=n_classes)

data_df = pd.read_csv('eval_segments_file_train.csv')
train_df = pd.DataFrame(columns=['filename', 'labels'])
for idx, row_data in tqdm(data_df.iterrows()):
    try:
        extract = pd.DataFrame(columns=['audio_name', 'feature', 'labels'])
        data_item = {}
        waveform, sr = read_wav(row_data['filename'])

        waveform = np.reshape(waveform, [1, -1])
        spectrogram = features_lib.waveform_to_log_mel_spectrogram(
                    tf.squeeze(waveform, axis=0), params)
        features = features_lib.spectrogram_to_patches(spectrogram, params)
        labels = []
        for i in range(features.shape[0]):
            labels.append(row_data['labels'])
        labels = [ast.literal_eval(i) for i in labels]
        labels = encoder_label().fit_transform(labels)
        #print(features.shape, labels.shape)
        data_item['audio_name'] = row_data['filename']
        data_item['feature'] = features
        data_item['labels'] = labels
        extract = extract.append(data_item,ignore_index=True)
        #break
        extract.to_hdf('eval_feature/' + row_data['filename'].split('/')[-1].split('.')[0] + '.h5', 'towdata')
        data_train = {}
        data_train['filename'] = row_data['filename']
        data_train['labels'] = row_data['labels']
        train_df = train_df.append(data_train,ignore_index=True)
    except:
        print(row_data['filename'])
        
train_df.to_csv('eval_data_train.csv', index=False)
# %%
# input_data = pd.read_hdf('extract.h5')
# read_input = input_data.values
# print(read_input[0][0])
# print(read_input[0][1].shape)
# print(read_input[0][2].shape)

# %%
