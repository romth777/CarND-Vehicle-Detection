import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
from collections import deque

import ImageProcessUtils
import Model


class Pipeline:
    def __init__(self):
        self.ipu = ImageProcessUtils.ImageProcessUtils()
        self.model = Model.Model()
        self.isfound = False
        self.img_deque = deque(maxlen=5)

    # apply gradient using some method and combine them in one image
    def gradient_image(self, img):
        gradx = self.ipu.abs_sobel_thresh(img, orient='x', sobel_kernel=17, thresh_min=25, thresh_max=170)
        grady = self.ipu.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh_min=25, thresh_max=220)
        mag_binary = self.ipu.mag_thresh(img, sobel_kernel=7, mag_thresh=(10, 130))
        dir_binary = self.ipu.dir_threshold(img, sobel_kernel=11, thresh=(0.5, 1.4))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) | (grady == 1) | (mag_binary == 1)) & (dir_binary == 1)] = 1
        return combined

    # combine the signal of R and S with thresholds
    def clearlify_image(self, img):
        s_binary = self.ipu.hls_select(img, thresh=(80, 255), channel=2)
        r_binary = self.ipu.rgb_select(img, thresh=(220, 250), channel=0)

        combined = np.zeros_like(r_binary)
        combined[((s_binary == 1) | (r_binary == 1))] = 1
        return combined

    # imput image is not warped
    def mix_image_of_grad_clearlify(self, img):
        grad = self.ipu.warp(self.gradient_image(img))
        origin = self.clearlify_image(self.ipu.warp(img))

        combined = np.zeros_like(grad)
        combined[((grad > 0) | (origin == 1))] = 1
        return combined

    # reset function clear the data of previous analysis
    def reset(self):
        self.left_line.reset()
        self.right_line.reset()
        self.isfound = False
        self.img_deque.clear()

    # Main pipeline process of this project
    def pipeline(self, img):
        # convert dtype for uint8 for processing
        img = img.astype(np.uint8)

        # sliding window
        windows = self.ipu.slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        windowed_img = self.ipu.draw_boxes(img, windows, color=(0, 0, 255), thick=6)

        # ditect car with classifier and record the position of window
        svc = self.model.get_clf()
        X_scaler = self.model.get_scaler()
        ystart = 400
        ystop = 656
        scale = 1.5
        out_img = self.find_cars(img, ystart, ystop, scale, svc, X_scaler, self.model.orient, self.model.pix_per_cell,
                                 self.model.cell_per_block, self.model.spatial_size, self.model.hist_bins)
        plt.imshow(out_img)
        plt.show()

        # stitch windows to centeroid and filter out false positive with heatmap

        # filter out false positive in one frame, not in the next frame


        return result

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                  hist_bins):

        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = self.ipu.convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self.ipu.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = self.ipu.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = self.ipu.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = self.ipu.bin_spatial(subimg, size=spatial_size)
                hist_features = self.ipu.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

        return draw_img

    # show image as arranged
    def arrange_images(self, img1, img2, title1, title2):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img1)
        ax1.set_title(title1, fontsize=25)
        ax2.imshow(img2)
        ax2.set_title(title2, fontsize=25)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    # run the main pipeline for video
    def run_pipeline(self, video_file, duration=None, end=False):
        """Runs pipeline on a video and writes it to temp folder"""
        print('processing video file {}'.format(video_file))
        clip = VideoFileClip(video_file)

        if duration is not None:
            if end:
                clip = clip.subclip(clip.duration - duration)
            else:
                clip = clip.subclip(0, duration)

        fpath = 'temp/' + video_file
        if os.path.exists(fpath):
            os.remove(fpath)
        processed = clip.fl(lambda gf, t: self.pipeline(gf(t)), [])
        processed.write_videofile(fpath, audio=False)


def main():
    pl = Pipeline()

    do_images = True
    do_videos = False

    if do_images:
        images = glob.glob('test_images/test1.jpg')

        for fname in images:
            image = mpimg.imread(fname)
            output = pl.pipeline(image)
            mpimg.imsave(os.path.join('output_images', 'output_' + os.path.basename(fname)), output)
            plt.imshow(output)
            pl.reset()

    if do_videos:
        video_files = ['project_video.mp4']
        for video_file in video_files:
            pl.run_pipeline(video_file)
            pl.reset()

main()
