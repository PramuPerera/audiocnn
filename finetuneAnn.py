'''
Edits made by Pramuditha
'''
import scipy.io
import tensorflow as tf
import numpy
import json
import cv2
import librosa
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import model_from_json
#import keras_resnet.models
from keras.models import Model
import pylab
from keras import backend as K
from tensorflow.python.client import device_lib
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
K.tensorflow_backend._get_available_gpus()
print(device_lib.list_local_devices())
# fix random seed for reproducibility
numpy.random.seed(7)
VERBOSE = False


def extract_data(data_path, core, data_length=None, batch=None, model='audiocnn'):
    """ Currently have 17 anomalies """

    if VERBOSE:
        print("getting data from json file")

    if data_length:
        start_loc = batch * data_length
        stop_loc = start_loc + data_length
        if stop_loc > len(core):
            stop_loc = len(core)

        sounds = core[start_loc:stop_loc:1]

    else:
        sounds = core

    # normal and anomaly label and feature vectors
    numpy.random.shuffle(sounds)

    labels = []
    fvecs = []
    fnames = {}
    if VERBOSE:
        print('Looping through json array')

    count = 1
    types = { "Normal":[0, 0, 0, 1], "normal":[0, 0, 0, 1]}
    for sound in sounds:

        s_type = sound['type']
        if s_type in types:
            labels.append(types[s_type])

        else:
	
            labels.append([1,0,0,0])
	    print('s')

        # open random image of dimensions 639x516
        if VERBOSE:
            print( "DataPath: ", data_path, " Image: ", sound['image'], " Type: ", s_type )
	
        img = cv2.imread(data_path + '/' + sound['image'], 0)
        #npy_arr = numpy.load(data_path + '/' + sound[0]['image'])
	if model == 'audiocnn':
	        if img.shape == (513, 800):
	            # put image in 4D tensor of shape (1, height, width, 1)
	            #img_ = img.transpose(2, 0, 1)
	            #npy_arr_ = npy_arr.transpose(2, 0, 1)
	            img_ = img.reshape(img.shape[0], img.shape[1], 1)

	            # This One fvecs.append(img_)
	            fvecs.append(img_)
	elif model == 'resnet':
		    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		    img_ = cv2.resize(img, (224,224))
		    fvecs.append(img_)

    fvecs = numpy.array(fvecs).astype(numpy.float32)

    split = fvecs.shape[0] - (fvecs.shape[0] // 5)

    xtrain= fvecs

    ytrain=labels
    # convert the int numpy array into a one-hot matrix
    num_classes = len(types.keys())

    return xtrain, ytrain, num_classes

def test(filename,outname, k=6, dim=4, model='audiocnn'):
	scores = {}
	data_path = filename #'/home/labuser/AudioCNN/new data/annotated/mic0/combined/' #pettijohncnormal/mic0/combined'
	core = json.load(open(data_path + '/labels.json'))
	if model=='audiocnn':
		json_file = open('original/four_categories.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("original/four_categories.h5")
		loaded_model.summary()
		if dim ==64:
			loaded_model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer('activation_4').output)
		# evaluate loaded model on test data
		loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
		print("Loaded model from disk")
		img_shape=(513, 800, 1)
		test_labels, test_data, num_classes = extract_data(data_path, core, model='audiocnn') #data and labels are flippe


	elif model=='resnet':
		img_shape=(224, 224, 1)
		print("Load the Data")
		loaded_model = ResNet50(weights='imagenet', pooling=max, include_top = True)
		# load weights into new model
		loaded_model.summary()
		#loaded_model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer('activation_4').output)
		print("Loaded model from disk")
		# evaluate loaded model on test data
		loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
		test_labels, test_data, num_classes = extract_data(data_path, core, model='resnet') #data and labels are flippe

	score = numpy.zeros((len(test_data), dim))
	data_length = 1000
	print("Number of samples: " + str(len(test_data)))
	for i in range(0,int(len(test_data)/k)):
		d = test_labels[i*k:k*(i+1)][:][:][:]
		temp = loaded_model.predict(d)
		score[i*k:k*(i+1)][:] = temp
	scores['score'] = score
	scores['class'] = test_data
	scipy.io.savemat(outname, scores)
