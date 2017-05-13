import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

import ImageProcessUtils

ipu = ImageProcessUtils.ImageProcessUtils()


class Model:
    def __init__(self):
        self.load_fname = 'hoglinearsvc.pkl'
        self.save_fname = self.load_fname
        self.save_scaler_fname = 'scaler.pkl'

        self.image_size = (64, 64)
        self.color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 14  # HOG orientations
        self.pix_per_cell = 16  # HOG pixels per cell
        self.cell_per_block = 3  # HOG cells per block
        self.hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (8, 8)  # Spatial binning dimensions
        self.hist_bins = 16  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off

    def train(self):
        cars = glob.glob(os.path.join("training_images", "vehicles", "**", "*.png"), recursive=True)
        notcars = glob.glob(os.path.join("training_images", "non-vehicles", "**", "*.png"), recursive=True)

        #sample_size = 500
        #cars = cars[0:sample_size]
        #notcars = notcars[0:sample_size]

        t = time.time()
        print("Start hogging to cars")
        car_features = []
        for car_fname in cars:
            car = mpimg.imread(car_fname)
            car = car.astype(np.float32) / 255
            car_features.append(ipu.single_img_features(car, color_space=self.color_space,
                                            spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                            orient=self.orient, pix_per_cell=self.pix_per_cell,
                                            cell_per_block=self.cell_per_block,
                                            hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                            hist_feat=self.hist_feat, hog_feat=self.hog_feat))
        print("Start hogging to not cars")
        notcar_features = []
        for notcar_fname in notcars:
            notcar = mpimg.imread(notcar_fname)
            notcar = notcar.astype(np.float32) / 255
            notcar_features.append(ipu.single_img_features(notcar, color_space=self.color_space,
                                               spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                               orient=self.orient, pix_per_cell=self.pix_per_cell,
                                               cell_per_block=self.cell_per_block,
                                               hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                               hist_feat=self.hist_feat, hog_feat=self.hog_feat))

        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        joblib.dump(X_scaler, self.save_scaler_fname)

        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        parameters = {'C': [0.001]} # 0.001 is the best
        # parameters = {'C': [x / 1000.0 for x in range(1, 11, 1)]}

        # Use a linear SVC
        svc = LinearSVC(random_state=1)

        clf = GridSearchCV(svc, parameters)
        # Check the training time for the SVC
        t = time.time()
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        joblib.dump(clf.best_estimator_, self.save_fname)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    def get_clf(self):
        return joblib.load(self.load_fname)

    def get_scaler(self):
        return joblib.load(self.save_scaler_fname)


if __name__ == '__main__':
    do_train = True
    do_predict = False

    model = Model()
    if do_train:
        model.train()

    if do_predict:
        cars = glob.glob(os.path.join("training_images", "vehicles", "**", "*.png"), recursive=True)
        notcars = glob.glob(os.path.join("training_images", "non-vehicles", "**", "*.png"), recursive=True)

        sample_size = 500
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

        t = time.time()
        print("Start hogging to cars")
        car_features = []
        for car_fname in cars:
            car = mpimg.imread(car_fname)
            car = car.astype(np.float32) / 255
            car_features.append(ipu.single_img_features(car, color_space=model.color_space,
                                            spatial_size=model.spatial_size, hist_bins=model.hist_bins,
                                            orient=model.orient, pix_per_cell=model.pix_per_cell,
                                            cell_per_block=model.cell_per_block,
                                            hog_channel=model.hog_channel, spatial_feat=model.spatial_feat,
                                            hist_feat=model.hist_feat, hog_feat=model.hog_feat))
        print("Start hogging to not cars")
        notcar_features = []
        for notcar_fname in notcars:
            notcar = mpimg.imread(notcar_fname)
            notcar = notcar.astype(np.float32) / 255
            notcar_features.append(ipu.single_img_features(notcar, color_space=model.color_space,
                                               spatial_size=model.spatial_size, hist_bins=model.hist_bins,
                                               orient=model.orient, pix_per_cell=model.pix_per_cell,
                                               cell_per_block=model.cell_per_block,
                                               hog_channel=model.hog_channel, spatial_feat=model.spatial_feat,
                                               hist_feat=model.hist_feat, hog_feat=model.hog_feat))
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        clf = model.get_clf()

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
