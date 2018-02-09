'''
Extract a known face from a video or image sequence.

This class uses a combination of a deep learning CNN model to batch detect faces
in video frames (in GPU with CUDA) then HoG to compare faces with a pre existing
reference set of face encodings.
'''

from os import path, listdir
from tqdm import tqdm

import cv2
import numpy
import face_recognition

class FaceGrab():
    '''Holds common settings for the reference encodings and processing parameters.
    so that multiple videos can be processed against them'''
    def __init__(self, reference, batch_size=32, skip_frames=6):
        self.batch_size = numpy.clip(batch_size, 2, 128)
        self.skip_frames = 0 if skip_frames < 0 else skip_frames + 1
        self.process_frames = []
        self.orignal_frames = []
        self.reference_encodings = []
        self.detected = 0
        print('Checking references...')
        # reference could be a single image or a directory of images
        # in either case we need the encoding data from the image(s)
        if path.isdir(reference):
            for file in listdir(reference):
                self.__parse_encodings(path.join(reference, file))
        else:
            self.__parse_encodings(reference)
        if len(self.reference_encodings) == 0:
            print('Warning: no references have been detected')
            print('if you process a video *all* matching faces will be extracted')
    def __parse_encodings(self, image_path):
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if numpy.any(encoding):
            self.reference_encodings.append(encoding[0])
            print('Found ref #{} in {}'.format(len(self.reference_encodings), image_path))

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
                         "tsnrhtdd"[(number / 10 % 10 != 1) * 
                                    (number % 10 < 4) * number % 10::4])

    def __recognise(self, encoding):
        '''Checks the unknown_encoding exits and compares against the known encoding(s).
        If no encodings at all are present then all faces are recognised'''
        if len(self.reference_encodings) == 0:
            return True
        if numpy.any(encoding):
            return numpy.any(face_recognition.compare_faces(self.reference_encodings, encoding[0]))
        return False

    def __batch(self, frame_count, output_directory):
        '''Finds all faces in batch of frames using precomputed CNN model (in GPU)
        Then checks all the found faces against a known reference encoding.
        If there is a match it saves the found face to the output directory'''
        location_sets = face_recognition.batch_face_locations(self.process_frames, batch_size=self.batch_size)
        for position, locations in enumerate(tqdm(iterable=location_sets, desc='Current batch')):
            frame = frame_count - self.batch_size + position
            for face_number, face_location in enumerate(locations):
                face = self.__extract(self.orignal_frames[position], face_location)
                if self.__recognise(face_recognition.face_encodings(face)):
                    self.detected += 1
                    cv2.imwrite(path.join(output_directory, '{}-{}-{}.jpg'.format(frame, position, face_number)), face)
                    # each frame v.unlikely to contain desired face more than once - break
                    # for speed.
                    break

    def process(self, input_video, output_directory='.'):
        '''Opens a video file and hands of batches off frames from it for processing'''
        video_capture = cv2.VideoCapture(input_video)
        if not video_capture.isOpened():
            raise Exception('Could not open the video file {}'.format(input_video))
        frame_count = 0
        self.detected = 0
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Opening {}...'.format(input_video))
        print('Processing every {} frame of {} frames'.format(self.__ordinalth(self.skip_frames),
                                                              total_frames))
        print('Checking each found face against {} reference(s)'.format(len(self.reference_encodings)))
        with tqdm(desc='Total frames', total=total_frames) as progress_main:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break
                frame_count += 1
                progress_main.update(1)
                if self.skip_frames > 0 and not frame_count % self.skip_frames:
                    continue
                self.process_frames.append(self.__downsample(frame))
                self.orignal_frames.append(frame)
                if len(self.process_frames) == self.batch_size:
                    self.__batch(frame_count, output_directory)
                    self.process_frames = []
                    self.orignal_frames = []
        progress_main.close()
        print('Found and extracted {} images in {}'.format(self.detected, output_directory))

if __name__ == '__main__':
    # Just for example...
    OUTPUT_DIR = r'.\output'
    REF_DIR = r'.\input\reference'
    REF_IMG = r'.\input\ref.jpg'
    TEST_VIDEO = r'.\input\vid.mp4'
    # reference can be a path to a single file (e.g. .\images\me.jpg)
    # or a path to an directory containing multiple images sequence (e.g. .\images)
    FG = FaceGrab(reference=REF_DIR)
    # input_video can be a path to a single file (eg.  D:\video\foo.mp4)
    # or a path to an image sequence (e.g.  D:\frames\img_%04d.jpg)
    # which will read image like img_0000.jpg, img_0001.jpg, img_0002.jpg, ...)
    FG.process(input_video=TEST_VIDEO, output_directory=OUTPUT_DIR)
