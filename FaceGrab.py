'''
Extract a known face from a video.

This class uses a combination of a deep learning CNN model to batch detect faces
in video frames, or a sequence of images in GPU with CUDA.
It then uses HoG to compare the deteted faces with a pre existingreference set of face encodings.
'''

from os import path, listdir
from tqdm import tqdm

import cv2
import numpy
import face_recognition

class FaceGrab():
    '''Holds common settings for the reference encodings and processing parameters.
    so that multiple videos can be processed against them'''
    def __init__(self, reference, batch_size=32, skip_frames=5, tolerance=.5):
        self.batch_size = numpy.clip(batch_size, 2, 128)
        self.skip_frames = 0 if skip_frames < 0 else skip_frames + 1
        self.tolernace = numpy.clip(tolerance, .1, 1)
        self._process_frames = []
        self._orignal_frames = []
        self._reference_encodings = []
        self._total_extracted = 0
        # reference could be a single image or a directory of images
        # in either case we need the encoding data from the image(s)
        if path.isdir(reference):
            for file in listdir(reference):
                self.__parse_encoding(path.join(reference, file))
        elif path.isfile(reference):
            self.__parse_encoding(reference)
        if not self._has_encodings:
            print('Warning: no references have been detected')
            print('Are you sure the reference path is correct? {}'.format(reference))
            print('If you process a video *all* detected faces will be extracted')

    def __parse_encoding(self, image_path):
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if numpy.any(encoding):
            self._reference_encodings.append(encoding[0])
            print('Found ref #{} in {}'.format(len(self._reference_encodings), image_path))

    @property
    def _has_encodings(self):
        return numpy.any(self._reference_encodings)

    @staticmethod
    def __downsample(frame):
        '''Downscale frame of video by 4 for faster recognition processing.
        also converts cv2's BGR to face_recognition's RGB'''
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = small_frame[:, :, ::-1] # BGR->RGB for detect (fuck I love numpy)
        return small_frame

    @staticmethod
    def __extract(frame, location):
        '''Upscale coords in face_location by 4 and extract from frame'''
        top, right, bottom, left = location
        return frame[top * 4:bottom * 4, left * 4:right * 4] # I mean it is awsome

    @staticmethod
    def __ordinalth(number):
        '''Given a number returns it suffixed with the ordinal string.
        i.e. 1=1st, 2=2nd, 3=3rd, etc'''
        return "%d%s" % (number,
                         "tsnrhtdd"[(number / 10 % 10 != 1) * (number % 10 < 4) * number % 10::4])

    def __recognise(self, encoding):
        '''Checks the unknown_encoding exits and compares against the known encoding(s).
        If no encodings at all are present then all faces are recognised.'''
        if not self._has_encodings:
            return True
        if numpy.any(encoding):
            return numpy.any(face_recognition.compare_faces(self._reference_encodings,
                                                            encoding[0],
                                                            self.tolernace))
        return False

    def __batch(self, batch_count, frame_count, output_directory):
        '''Finds all faces in batch of frames using precomputed CNN model (in GPU)
        Then checks all the found faces against a set of known reference encodings.
        If there is a match it saves the found face to the output directory'''
        location_sets = face_recognition.batch_face_locations(self._process_frames,
                                                              batch_size=self.batch_size)
        extracted = 0
        with tqdm(total=len(location_sets)) as progress_batch:
            for position, locations in enumerate(location_sets):
                frame = frame_count - self.batch_size + position
                progress_batch.update(1)
                progress_batch.set_description('Batch #{} (recognised {})'.format(batch_count,
                                                                                  extracted))
                for face_number, face_location in enumerate(locations):
                    face = self.__extract(self._orignal_frames[position], face_location)
                    if self.__recognise(face_recognition.face_encodings(face)):
                        extracted += 1
                        self._total_extracted += 1
                        output_path = path.join(output_directory,
                                                '{}-{}-{}.jpg'.format(frame,
                                                                      position,
                                                                      face_number))
                        cv2.imwrite(output_path, face)
                        # frame v.unlikely to contain desired face more than
                        # once
                        break

    def process(self, input_video, output_directory='.'):
        '''Opens a video file and hands of batches off frames from it for processing'''
        video_capture = cv2.VideoCapture(input_video)
        if not video_capture.isOpened():
            raise Exception('Could not open the video file {}'.format(input_video))
        frame_count = 0
        batch_count = 0
        self._total_extracted = 0
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_batches = int(total_frames / self.skip_frames / self.batch_size)
        refrence_count = len(self._reference_encodings)
        print('Processing {}...'.format(input_video))
        print('every {} frame of {} for {} batches'.format(self.__ordinalth(self.skip_frames),
                                                           total_frames,
                                                           total_batches))
        print('Checking detected against {} reference{}'.format(refrence_count,
                                                                's' if refrence_count > 1 else ''))
        with tqdm(total=total_frames) as progress_main:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break
                frame_count += 1
                progress_main.update(1)
                progress_main.set_description('Total (extracted {})'.format(self._total_extracted))
                if self.skip_frames > 0 and frame_count % self.skip_frames:
                    continue
                self._process_frames.append(self.__downsample(frame))
                self._orignal_frames.append(frame)
                if len(self._process_frames) == self.batch_size:
                    batch_count += 1
                    self.__batch(batch_count, frame_count, output_directory)
                    self._process_frames = []
                    self._orignal_frames = []
        progress_main.close()
        print('\nFound and grabbed {} faces'.format(self._total_extracted))
        
if __name__ == '__main__':
    # Just for example...
    OUTPUT_DIR = r'.\output-files'
    REF_DIR = r'.\reference-images'
    TEST_VIDEO = r'.\input\video.mp4'
    # reference can be a path to a single file (eg.  D:\images\someone.jpg)
    # or a path to an directory an images sequence (e.g.  D:\images)
    FG = FaceGrab(reference=REF_DIR, batch_size=50, skip_frames=25)
    # input_video can be a path to a single file (eg.  D:\video\foo.mp4)
    # or a path to an image sequence (e.g.  D:\frames\img_%04d.jpg)
    # which will read image like img_0000.jpg, img_0001.jpg, img_0002.jpg, ...)
    FG.process(input_video=TEST_VIDEO, output_directory=OUTPUT_DIR)
