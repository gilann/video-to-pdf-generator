from flask import Flask, render_template, request, send_file
app = Flask(__name__)
import cv2
import numpy as np
import os
import sys
import math
import copy
from PIL import Image

__hist_size__ = 128             # how many bins for each R,G,B histogram
# if a shot has length less than this, merge it with others
__min_duration__ = 25
__absolute_threshold__ = 1.0   # any transition must be no less than this threshold


class shotDetector(object):
    def __init__(self, video_path=None, min_duration=__min_duration__, output_dir=None):
        self.video_path = video_path
        self.min_duration = min_duration
        self.output_dir = output_dir
        self.factor = 6

    def run(self, video_path=None):
        if video_path is not None:
            self.video_path = video_path
        assert (self.video_path is not None), "you should must the video path!"

        self.shots = []
        self.scores = []
        self.frames = []
        hists = []
        cap = cv2.VideoCapture(self.video_path)
        self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            success, frame = cap.read()
            if not success:
                break
            self.frames.append(frame)
#            millis = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
#            print millis
            # compute RGB histogram for each frame
            chists = [cv2.calcHist([frame], [c], None, [__hist_size__], [0, 256])
                      for c in range(3)]
            chists = np.array([chist for chist in chists])
            hists.append(chists.flatten())
        # compute hist  distances
        self.scores = [np.ndarray.sum(abs(pair[0] - pair[1]))
                       for pair in zip(hists[1:], hists[:-1])]

    def pick_frame(self, obj_path=None, video_split_id=None):
        average_frame_div = sum(self.scores)/len(self.scores)
        self.obj_path = obj_path
        self.frame_index = []
        self.video_split_id = video_split_id
        for idx in range(len(self.scores)):
            if self.scores[idx] > self.factor * average_frame_div:
                self.frame_index.append(idx + 1)

        tmp_idx = copy.copy(self.frame_index)
        for i in range(0, len(self.frame_index) - 1):
            if self.frame_index[i + 1] - self.frame_index[i] < __min_duration__:
                del tmp_idx[tmp_idx.index(self.frame_index[i])]
        self.frame_index = tmp_idx
        print("special frames have {0}".format(len(self.frame_index)))

        if self.video_split_id and self.obj_path:
            # the real index start from 1 but time 0 and end add to it
            idx_new = copy.copy(self.frame_index)
            idx_new.insert(0, -1)
            if len(self.frames) - 1 - idx_new[-1] < __min_duration__:
                del idx_new[-1]
            idx_new.append(len(self.frames) - 1)

            idx_new = list(map(lambda x: x + 1.0, idx_new))
            frame_middle_idx = [math.ceil((pair[0] + pair[1])/2)
                                for pair in zip(idx_new[:-1], idx_new[1:])]
            frame_middle_idx = list(map(lambda x: int(x), frame_middle_idx))
            #time_idx = map(lambda x : x / self.fps, idx_new)
            #timestamp_index = [(pair[0], pair[1]) for pair in zip(time_idx[:-1], time_idx[1:])]
            #print idx_new
            #print timestamp_index
            #print frame_middle_idx

            os.system("mkdir -p {0}".format(self.obj_path))
            tmp_idx = copy.copy(frame_middle_idx)
            for i in range(0, len(frame_middle_idx) - 1):
                if frame_middle_idx[i + 1] - frame_middle_idx[i] < __min_duration__:
                    del tmp_idx[tmp_idx.index(frame_middle_idx[i])]
            frame_middle_idx = tmp_idx
            print(frame_middle_idx)
            print(idx_new)
            time_idx = [0.0]
            #frame_idx_tmp = map(lambda x : x + 1, frame_middle_idx)
            frame_idx_tmp = frame_middle_idx
            for i, element in enumerate(frame_idx_tmp):
                if i < len(frame_idx_tmp) - 1:
                    time_point = list(
                        filter(lambda x: x <= frame_idx_tmp[i + 1], idx_new))[-1]
                    if time_point not in time_idx:
                        time_idx.append(time_point)
                # elif idx_new[-1] - time_idx[-1] < __min_duration__:
                else:
                    #del time_idx[-1]
                    time_idx.append(idx_new[-1])
#            for element in frame_middle_idx:
#                time_point = filter(lambda x: x > element, idx_new)[0]
#                time_idx.append(time_point)
            print(time_idx)
            time_idx_float = list(map(lambda x: x / self.fps, time_idx))
            print(time_idx_float)
            timestamp_index = [(pair[0], pair[1]) for pair in zip(
                time_idx_float[:-1], time_idx_float[1:])]

            for i, idx in enumerate(frame_middle_idx):
                print(i, idx)
                cv2.imwrite("{0}/{1}{2:.1f}-{3:.1f}.jpg".format(self.obj_path, self.video_split_id,
                                                                timestamp_index[i][0], timestamp_index[i][1]), self.frames[idx - 1])

        else:
            for idx in self.frame_index:
                if self.obj_path is None:
                    print("hello")
                    cv2.imwrite("{0}.jpg".format(idx + 1), self.frames[idx])

                else:
                    os.system("mkdir -p {0}".format(self.obj_path))
                    cv2.imwrite(
                        "{0}/frame{1}.jpg".format(self.obj_path, idx + 1), self.frames[idx])


@app.route('/upload')
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def videoToImage():
    if request.method == 'POST':
        f = request.files['file']
        f.save((f.filename))

        factor = 6
        detector = shotDetector(f.filename)
        detector.run()
        detector.pick_frame("/home/shivam/Desktop/project/picOut", "images")
        path = '/home/shivam/Desktop/project/picOut'
        files = []
        images = []
        imageList = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in sorted(f, reverse=True):
                if '.jpg' in file:
                    files.append(os.path.join(r, file))
        files.sort()
        for f in files:
            images.append(Image.open(f))
            print(f)
        for image in images:
            imageList.append(image.convert('RGB'))

        imageList.pop(0).save(r'/home/shivam/Desktop/project/output/myNewImages.pdf',
                              save_all=True, append_images=imageList)
        for f in files:
            os.remove(f)
        
        return send_file('/home/shivam/Desktop/project/output/myNewImages.pdf', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
