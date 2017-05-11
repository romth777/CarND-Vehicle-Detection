import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from skimage.feature import hog


class ImageProcessUtils:
    def __init__(self):
        # image size for this project, this value must change in the other image shape
        self.img_size = (1280, 720)

    # Define a function to compute binned color features
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # Define a function to compute color histogram features
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def convert_color(self, img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
        # 1) Define an empty list to receive features
        img_features = []
        # 2) Apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)
        # 3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            # 4) Append features to list
            img_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            # 6) Append features to list
            img_features.append(hist_features)
        # 7) Compute HOG features if flag is set
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
            else:
                hog_features = self.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        return np.concatenate(img_features)

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32),
                         hist_bins=32, orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
                         spatial_feat=True, hist_feat=True, hog_feat=True):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                feature_image = np.copy(image)

            if spatial_feat == True:
                spatial_features = bin_spatial(feature_image, size=spatial_size)
                file_features.append(spatial_features)
            if hist_feat == True:
                # Apply color_hist()
                hist_features = color_hist(feature_image, nbins=hist_bins)
                file_features.append(hist_features)
            if hog_feat == True:
                # Call get_hog_features() with vis=False, feature_vec=True
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                             orient, pix_per_cell, cell_per_block,
                                                             vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features

    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def hls_select(self, img, thresh=(0, 255), channel=2):
        """Return a channel from HLS"""
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
        selected_channel = hls[:, :, channel]
        binary = np.zeros_like(selected_channel)
        binary[(selected_channel > thresh[0]) & (selected_channel <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        binary_output = np.copy(binary)  # placeholder line
        return binary_output

    def rgb_select(self, img, thresh=(0, 255), channel=2):
        """Return a channel from RGB"""
        # 2) Apply a threshold to the selected channel
        selected_channel = img[:, :, channel]
        binary = np.zeros_like(selected_channel)
        binary[(selected_channel > thresh[0]) & (selected_channel <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        binary_output = np.copy(binary)  # placeholder line
        return binary_output

    # Define a function that applies Sobel x or y,
    # then takes an absolute value and applies a threshold.
    # Note: calling your function with orient='x', thresh_min=5, thresh_max=100
    # should produce output like the example image shown above this quiz.
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh_min=20, thresh_max=100):
        """Apply single direction sobel filter and threshold of them"""
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = hls[:, :, 2]

        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the magnitude of the gradient
    # and applies a threshold
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(30, 100)):
        """Apply mixed direction sobel filter with sqrt(sobelx ** 2 + sobely ** 2) and threshold of them"""
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = hls[:, :, 2]

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Calculate the magnitude
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradmag = np.absolute(sobel)

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.int8)

        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(self, img, sobel_kernel=15, thresh=(0.8, 1.2)):
        """Apply mixed direction sobel filter with arctan2(sobelx, sobely) and threshold of them"""
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = hls[:, :, 2]

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the derivative or gradient
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        atan_sobel = np.arctan2(abs_sobely, abs_sobelx)

        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(atan_sobel)
        binary_output[(atan_sobel >= thresh[0]) & (atan_sobel <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return binary_output
