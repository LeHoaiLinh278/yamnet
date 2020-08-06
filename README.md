# Train YAMNET

<!-- GETTING STARTED -->
## Getting Started
YAMNet is a pretrained deep net that predicts 521 audio event classes based on the [AudioSet-YouTube corpus](http://g.co/audioset)

### Prerequisites
```shell
# install tensorflow for cuda 10.1
sudo pip3 install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.2.0-cp36-cp36m-manylinux2010_x86_64.whl 
# install 
sudo pip3 install -r requirement.txt
```
## About the Model
The YAMNet code layout is as follows:
* `params.py`: Hyperparameters.  You can usefully modify PATCH_HOP_SECONDS.
* `features.py`: Audio feature extraction helpers.
* `save_file_csv.py` : Create list file audio and label, save file csv
* `utils.py` : Read file wav
* `extract_h5.py` : Extract feature file wav to vector and save file csv used to train
