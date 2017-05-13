import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

import ImageProcessUtils
import Model


class Pipeline:
    def __init__(self):
        self.ipu = ImageProcessUtils.ImageProcessUtils()
        self.model = Model.Model()

    # reset function clear the data of previous analysis
    def reset(self):
        pass

    # Main pipeline process of this project
    def pipeline(self, img):
        # convert dtype for uint8 for processing
        img = img.astype(np.uint8)

        # sliding window
        xy_windows = [(64, 64), (96, 96), (128, 128)]
        windows = []
        for xy_window in xy_windows:
            windows.extend(self.ipu.slide_window(img, x_start_stop=[None, None], y_start_stop=[400, 656],
                         xy_window=xy_window, xy_overlap=(0.75, 0.75)))

        # ditect car with classifier and record the position of window
        svc = self.model.get_clf()
        X_scaler = self.model.get_scaler()
        car_windows = []
        count = 226
        for window in windows:
            cropped_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            cropped_img = cv2.resize(cropped_img, self.model.image_size)
            cropped_img = cropped_img.astype(np.float32) / 255
            features = self.ipu.single_img_features(cropped_img, color_space=self.model.color_space, spatial_size=self.model.spatial_size,
                                hist_bins=self.model.hist_bins, orient=self.model.orient,
                                pix_per_cell=self.model.pix_per_cell, cell_per_block=self.model.cell_per_block, hog_channel=self.model.hog_channel,
                                spatial_feat=self.model.spatial_feat, hist_feat=self.model.hist_feat, hog_feat=self.model.hog_feat)
            scaled_features = X_scaler.transform(features)
            pred = svc.predict(scaled_features)
            if pred == 1:
                car_windows.append(window)

        #windowed_img = self.ipu.draw_boxes(img, car_windows, color=(0, 0, 255), thick=6)

        # stitch windows to centeroid and filter out false positive with heatmap
        heatmap = np.zeros_like(img[:, :, 0])
        heat = self.ipu.add_heat(heatmap, car_windows)
        self.ipu.apply_threshold(heat, 15, 1)
        labels = label(heatmap)
        draw_img = self.ipu.draw_labeled_bboxes(img, labels)

        # filter out false positive in one frame, not in the next frame

        out_img = draw_img
        return out_img

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

    do_images = False
    do_videos = True

    if do_images:
        images = glob.glob('test_images/*.jpg')

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

if __name__ == '__main__':
    main()
