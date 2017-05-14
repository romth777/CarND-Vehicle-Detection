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
import HeatmapHistory

class Pipeline:
    def __init__(self):
        self.ipu = ImageProcessUtils.ImageProcessUtils()
        self.do_svc = True
        self.do_kb = False
        self.model = None
        self.annotate = None
        if self.do_svc:
            self.model = Model.Model()
        if self.do_kb:
            self.annotate = annotate.annotate()
        self.heatmap_history = HeatmapHistory.HeatmapHistory()

    # reset function clear the data of previous analysis
    def reset(self):
        self.heatmap_history.reset()

    # Main pipeline process of this project
    def pipeline_kb(self, img):
        # convert dtype for uint8 for processing
        img = img.astype(np.uint8)

        # apply kittibox
        out_img, pred_boxes = self.annotate.make_annotate(img, threshold=0.5)

        # windowed_img = self.ipu.draw_boxes(img, pred_boxes, color=(0, 0, 255), thick=6)
        # plt.imshow(windowed_img)
        # plt.show()

        # stitch windows to centeroid and filter out false positive with heatmap
        heatmap = np.zeros_like(img[:, :, 0])
        heat = self.ipu.add_heat(heatmap, pred_boxes)
        self.ipu.apply_threshold(heat, 1000, 3)
        labels = label(heatmap)
        draw_img = self.ipu.draw_labeled_bboxes(img, labels)

        out_img = draw_img
        return out_img

    # Main pipeline process of this project
    def pipeline_svc(self, img):
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
        for window in windows:
            cropped_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            cropped_img = cv2.resize(cropped_img, self.model.image_size)
            features = self.ipu.single_img_features(cropped_img, color_space=self.model.color_space, spatial_size=self.model.spatial_size,
                                hist_bins=self.model.hist_bins, orient=self.model.orient,
                                pix_per_cell=self.model.pix_per_cell, cell_per_block=self.model.cell_per_block, hog_channel=self.model.hog_channel,
                                spatial_feat=self.model.spatial_feat, hist_feat=self.model.hist_feat, hog_feat=self.model.hog_feat)
            scaled_features = X_scaler.transform(features)
            pred = svc.predict(scaled_features)
            if pred == 1:
                car_windows.append(window)

        # windowed_img = self.ipu.draw_boxes(img, car_windows, color=(0, 0, 255), thick=6)
        # plt.imshow(windowed_img)
        # plt.show()

        # stitch windows to centeroid and filter out false positive with heatmap
        heatmap = np.zeros_like(img[:, :, 0])
        heat = self.ipu.add_heat(heatmap, car_windows)
        heat_thresh = self.ipu.apply_threshold(heat, 100, 6)
        heat_mean = self.heatmap_history.update(heat_thresh)
        #plt.imshow(heat_mean, cmap='hot')
        #plt.show()
        labels = label(heat_mean)
        draw_img = self.ipu.draw_labeled_bboxes(img, labels)

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
        if self.do_svc:
            processed = clip.fl(lambda gf, t: self.pipeline_svc(gf(t)), [])
        if self.do_kb:
            processed = clip.fl(lambda gf, t: self.pipeline_kb(gf(t)), [])
        processed.write_videofile(fpath, audio=False)


def main():
    pl = Pipeline()

    do_images = False
    do_videos = True

    if pl.do_kb:
        from KittiBox import annotate

    if do_images:
        images = glob.glob('test_images/*.jpg')

        for fname in images:
            print(fname)
            image = mpimg.imread(fname)
            if pl.do_svc:
                output = pl.pipeline_svc(image)
            if pl.do_kb:
                output = pl.pipeline_kb(image)
            mpimg.imsave(os.path.join('output_images', 'output_' + os.path.splitext(os.path.basename(fname))[0] + ".png"), output)
            plt.imshow(output)
            pl.reset()

    if do_videos:
        video_files = ['project_video.mp4']
        #video_files = ['shorter_project_video.mp4']
        #video_files = ['oneshot_project_video.mp4']
        for video_file in video_files:
            pl.run_pipeline(video_file)
            pl.reset()

if __name__ == '__main__':
    main()
