# Writeup of Vehicle Detection
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/svc/output_test1.png
[image3]: ./output_images/svc/output_test6.png
[image4]: ./output_images/kb/output_test1.png
[image5]: ./output_images/kb/output_test6.png

[video1]: ./project_video.mp4

[gif1]: ./output_images/car_search_pix_per_cell.gif
[gif2]: ./output_images/car_ppc5_search_orient.gif
[gif3]: ./output_images/car_ppc8_search_orient.gif
[gif4]: ./output_images/notcar_sarch_pix_per_cell.gif
[gif5]: ./output_images/car_ppc5_search_cell_per_block.gif
[gif6]: ./output_images/car_ppc8_search_cell_per_block.gif
[gif7]: ./output_images/notcar_ppc8_search_cell_per_block.gif

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the "More sample images of car" code block of the IPython notebook of `./my_trial/my_trial.ipynb` (or in lines 101 through 110 of the file called `ImageProcessUils.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and change HOG parameters of `pixels_per_cell = (2, 2) to (8, 8)`:

![alt text][gif1]


#### 2. Explain how you settled on your final choice of HOG parameters.
I choose pixels_per_cell from above, and pick up `5 and 8`, because it seems like car silhouette.(I think, if human cannot detect shape of transfromed image, the machine could not detect at all.) And of course I thought that processing at the edge of image is good.

Next, I try to decide `orientations` like below:
#### search by pixels_per_cell = 5
![alt text][gif2]
#### search by pixels_per_cell = 8
![alt text][gif3]
#### compare with notcar data
![alt text][gif4]

In this case, I think that the point where convergence of the data converged is the value of a better variable, and from this figure it was set to `orientations=10 with pixels_per_cell=8` this time.

Finally, I check the cells_per_block variable like below:
#### search by pixels_per_cell = 5
![alt text][gif5]
#### search by pixels_per_cell = 8
![alt text][gif6]
#### compare with notcar data
![alt text][gif7]

Comparing with the notcar data, it seems like `cells_per_block=4` is the better choice of classification, I choose it.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using GridsearchCV for `C` value. It shows me `C=0.001` is the best answer for me.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at `x=0 to 1280`, `y=400 to 656`, `window_size=64, 96, 128`, and `overlap=0.75`. The value of y was decided from the width of the road on the angle of view. Window_size decided on the size of the car I want to detect from the width of field angle. Finally, overlap is overlooking, but this was set to a higher value because I wanted the location like car to concentrate at the time of the later heatmap processing.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image2]
![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/svc/result_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

And I created a heatmap that added the area enclosed in boxes. It takes a weighted average with multiple frames based on time series. Because it becomes impossible to trust as it becomes information of the past. Then they converted them by `scipy.ndimage.measurements.label()` And transformed them so that they can be handled as one block.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This time, I used the linear svm to detect cars, but the accuracy was a bit bad. Especially when it was animation, it appeared remarkably. I think that one of the reasons for this is overfitting. I believe that the image given this time and the data set used for training was influenced by light intensity and other environmental factors. Therefore, in order to obtain more accuracy, it is most effective to increase the number of training data sets. Also, if the data set can not be prepared, consider the position of the time series box and think that it is necessary to judge whether it is really a car or not. However, this is implemented to some extent in my code.  

Finally, I present the result using KittiBox as another approach to this problem. [KittiBox](https://github.com/MarvinTeichmann/KittiBox) is based on Fastbox(based on VGG16) pretrained with Kitti data. I prepared and implemented an interface that fits this project. The results are as follows.
![alt text][image4]
![alt text][image5]
Here's a [link to my video result](./output_images/kb/project_video.mp4)

Compared to this result clearly KittiBox seems to be able to detect car more accurately, but I believe it is due to the amount of training data set.
