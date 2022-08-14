import json
import numpy as np
import math 
import os
import cv2
from NeuralNetStructure.imagetools import ImageTools
class Preprocessor:


    def __init__(self, dataset_name, resolution):

        self.DATASET_NAME = dataset_name
        self.DATASET_PATH = os.path.join("Memory", self.DATASET_NAME)
        self.CLASSES_PATH = os.path.join(self.DATASET_PATH, "Classes")
        self.IMGBANK_PATH = os.path.join(self.DATASET_PATH, "ImgBank")
        self.JSON_PATH = os.path.join(self.IMGBANK_PATH, dataset_name + ".json")

        if not os.path.exists(self.IMGBANK_PATH):
            os.mkdir(self.IMGBANK_PATH)
        
        self.resolution = resolution
    

    def load_json_data(self, dims="1D"):

        labels = []
        self.ImgBank = {}
        if not os.path.exists(self.IMGBANK_PATH):
            with open(self.JSON_PATH, "r") as f:
                dataset = json.load(f)

            # Create labelmap with keys
            
            for label in dataset["labels"]:
                if label not in labels:
                    labels.append(label)
            for label in labels:
                self.ImgBank[label] = []
            
            # populate labelmap with values
            for index in range(len(dataset["labels"])):
                self.ImgBank[dataset["labels"][index]].append(dataset["data"][index])

            # Create labelmap json file
            if not os.path.exists(self.IMGBANK_PATH):
                with open(self.IMGBANK_PATH, "w") as labelmap_file:
                    json.dump(self.ImgBank, labelmap_file)
        else:
            with open(self.JSON_PATH, "r") as f:
                self.ImgBank = json.load(f)
        
        for label in self.ImgBank:
            for index in range(len(self.ImgBank[label])):
                self.ImgBank[label][index] = np.array(self.ImgBank[label][index])
                if dims=='2D':
                    self.ImgBank[label][index] = np.reshape(self.ImgBank[label][index], (3, 20, 20))
            
    def load_img_data(self):
        labels = []
        self.ImgBank = {}
        for class_folder in os.listdir(self.CLASSES_PATH):
            labels.append(class_folder)
            self.ImgBank[class_folder] = []
            for img in os.listdir(os.path.join(self.CLASSES_PATH, class_folder)):
                img = cv2.resize(cv2.imread(os.path.join(self.CLASSES_PATH, class_folder, img)), self.resolution)
                img = ImageTools.channelLastToFirst(img)
                img = ImageTools.normalize(img)
                self.ImgBank[class_folder].append(img)
    
    def save_img_data(self):
        with open(self.JSON_PATH, "w") as f:
            json.dump(self.ImgBank, f)

    def split(self, fraction):
        
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        # split each class into training and testing partitions
        for class_index, class_name in enumerate(self.ImgBank):

            class_size = len(self.ImgBank[class_name])
            train_range = math.floor(class_size * fraction)

            self.X_train.extend(self.ImgBank[class_name][:train_range])
            self.y_train.extend([class_index for i in range(train_range)])

            self.X_test.extend(self.ImgBank[class_name][train_range : class_size])
            self.y_test.extend([class_index for i in range(train_range, class_size) ])
        
        self.X_train = np.stack([data for data in self.X_train], 0)
        self.y_train = np.stack([data for data in self.y_train], 0)
        self.X_test = np.stack([data for data in self.X_test], 0)
        self.y_test = np.stack([data for data in self.y_test], 0)

    def shuffle(self):

        train_keys = np.array(range(np.shape(self.X_train)[0]))
        test_keys = np.array(range(np.shape(self.X_test)[0]))

        np.random.shuffle(train_keys)
        np.random.shuffle(test_keys)

        self.X_train = self.X_train[train_keys]
        self.X_test = self.X_test[test_keys]

        self.y_train = self.y_train[train_keys]
        self.y_test = self.y_test[test_keys]

    def preprocess(self, fraction, dims="1D"):
        if os.path.exists(self.JSON_PATH):
            self.load_json_data(dims=dims)
        else:
            self.load_img_data()
        self.split(fraction)
        self.shuffle()
   
        return self.X_train, self.y_train, self.X_test, self.y_test




