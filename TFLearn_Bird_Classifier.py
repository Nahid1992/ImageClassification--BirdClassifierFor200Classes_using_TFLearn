from __future__ import division, print_function, absolute_import
import tflearn
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.layers.normalization import local_response_normalization

import cv2
from PIL import Image

#Restarting karnel
print('Kernel Restarting..')
tf.reset_default_graph()
print('Kernel Restarted..')

img_height = 100
img_width = 100
nb_classes = 200

def dataLoad(I,L,B):
    #X,Y = tflearn.data_utils.image_preloader(dataFile,image_shape=(None,None),mode='folder',categorical_labels=True,normalize=True)
    #X,Y = tflearn.data_utils.shuffle(X,Y)

    imgList = []
    classList = []
    boxList = []

    with open(L) as f:
        for line in f:
            classList.append(line)
    with open(I) as f:
        for line in f:
            imgList.append(line)
    with open(B) as f:
        for line in f:
            boxList.append(line)

    X = []
    Y = []
    for i in range(0,len(imgList)):
        imgPath = imgList[i].split(' ')[-1].split('\n')[0]
        tempPath = imgPath
        imgPath = "C:/Users/Nahid/Documents/MachineLearningProjects/zDataset/CUB_200_2011/CUB_200_2011/images/" + imgPath
        original_img = Image.open(imgPath)

        check = np.array(original_img)
        check = len(check.shape)
        if check < 3:
            continue

        points = boxList[i].split(' ')
        pointA = int(points[1].split('.0')[0])
        pointB = int(points[2].split('.0')[0])
        pointC = int(points[3].split('.0')[0])
        pointD = int(points[4].split('.0')[0].split('\n')[0])
        crop_img = original_img.crop((pointA,pointB,pointC,pointD))
        rez_img = crop_img.resize((100,100), Image.ANTIALIAS)
        img = np.array(rez_img)
        X.append(img/255)
        Y.append(classList[i].split(' ')[1].split('\n')[0])
        print('Serial - ' + str(i) + ': ' + tempPath)

    #imgList = np.array(imgList)
    #labelList = np.array(labelList)
    #print('Size of ImageList = ' + str(imgList.shape))
    #print('Size of LabelList = ' + str(labelList.shape))
    Y = to_categorical(Y,201)
    Y = np.array(Y)
    X = np.array(X)
    #X = X.reshape(-1,img_width,img_height,3)
    '''
    x_train = X[:8250]
    x_val = X[8250:11788]
    y_train = Y[:8250]
    y_val = Y[8250:11788]
    '''
    '''
    x_train = X[:25]
    y_train = Y[:25]
    x_val = X[25:]
    y_val = Y[25:]
    '''
    X,Y = tflearn.data_utils.shuffle(X,Y)
    #output_path = 'data/_dataset.h5'
    #build_hdf5_image_dataset(dataset_file, image_shape=(img_width,img_height), mode='file', output_path=output_path, categorical_labels=True, normalize=True)
    return X,Y

def create_model():
	# Building 'AlexNet'
	network = input_data(shape=[None, img_width, img_height, 3])

	network = conv_2d(network, 96, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)

	network = conv_2d(network, 256, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)

	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)

	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)

	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)

	network = fully_connected(network, 60, activation='sigmoid')

	model = regression(network, optimizer='adam',
						 loss='categorical_crossentropy',
						 learning_rate=0.001)

	return model

def create_own_model():

    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Convolutional network building
    network = input_data(shape=[None, img_width, img_height, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, nb_classes+1, activation='softmax')

    model = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return model

def train_model(model,x_train,y_train):
	model = tflearn.DNN(model, tensorboard_verbose=3)

	model.fit(x_train, y_train, n_epoch=20,shuffle=False,
			show_metric=True, batch_size=100, snapshot_step=50,
			snapshot_epoch=False, run_id='tflean_bird_run01')

    #print('Model Trained...')
	#save Model
	model.save('models/tflearn_fruit_model_alexnet.model')
    #print('Model Saved...')

def load_model():
	model.load('models/tflearn_fruit_model_alexnet.model')

def main():
    dataFile = 'C:/Users/Nahid/Documents/MachineLearningProjects/zDataset/CUB_200_2011/CUB_200_2011/images/'
    bboxFile = 'C:/Users/Nahid/Documents/MachineLearningProjects/zDataset/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'
    imgList = 'C:/Users/Nahid/Documents/MachineLearningProjects/zDataset/CUB_200_2011/CUB_200_2011/images.txt'
    labelList = 'C:/Users/Nahid/Documents/MachineLearningProjects/zDataset/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
    x_train,y_train = dataLoad(imgList,labelList,bboxFile)
    print('Data Ready For Training...')

    print('TrainX = ' + str(len(x_train)))
    print('TrainY = ' + str(len(y_train)))
    print('ValX = ' + str(len(x_val)))
    print('ValY = ' + str(len(y_val)))

    #model = create_model()
    model = create_own_model()
    print('Model Created...')

    print('Training Started...')
    train_model(model,x_train,y_train)
    print('Model Trained & Saved...')
    print('Done...')

    breakPoint = 11
if __name__== "__main__":
  main()
