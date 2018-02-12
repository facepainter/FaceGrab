'''
Extract a known face from a video.

Uses a combination of a deep learning CNN model to batch detect faces
in video frames, or a sequence of images, in GPU with CUDA and HoG to compare
the detected faces with a computed reference set of face encodings.
'''

import argparse
from os import path, listdir
from typing import NamedTuple

import cv2
import numpy
import face_recognition
from tqdm import tqdm

class RecognitionSettings(NamedTuple):
    '''
    Face recognition settings
    :param float tolerance: How much "distance" between faces to consider it a match.
    :param int jitter: How many times to re-sample images when calculating encodings.
    '''
    tolerance: float = .6
    jitter: int = 10

class ProcessSettings(NamedTuple):
    '''
    Video process settings
    :param int batch_size: How many frames to include in each GPU processing batch.
    :param int skip_frames: How many frames to skip e.g. 5 means look at every 6th
    :param int extract_size: Size in pixels of extracted face images (n*n).
    :param float scale: Amount to down-sample input by for detection processing.
    '''
    batch_size: int = 128
    skip_frames: int = 6
    extract_size: int = 256
    scale: float = .25

class FaceGrab(object):
    '''
    It sure grabs faces! (tm)
        :param str reference: Path to a input data (video/image sequence)
        :param RecognitionSettings recognition: Face recognition settings
        :param ProcessSettings process: Video process settings
    '''
    def __init__(self, reference, recognition=None, process=None):
        if recognition is None:
            recognition = RecognitionSettings()
        if process is None:
            process = ProcessSettings()
        skip_sanity = 1 if process.skip_frames <= 0 else process.skip_frames + 1
        self._ps = process._replace(batch_size=numpy.clip(process.batch_size, 2, 128),
                                    skip_frames=skip_sanity,
                                    scale=numpy.clip(process.scale, 0, 1.0))
        self._rs = recognition._replace(tolerance=numpy.clip(recognition.tolerance, 0.1, 1))
        self._process_frames = []
        self._orignal_frames = []
        self._reference_encodings = []
        self._total_extracted = 0
        self.__check_reference(reference)

    @property
    def reference_count(self):
        '''Total currently loaded reference encodings for recognition'''
        return len(self._reference_encodings)

    @staticmethod
    def __downsample(image, scale):
        '''Downscale and convert image for faster detection processing'''
        sampled = cv2.resize(image, (0, 0), fx=scale, fy=scale) if scale > 0 else image
        return sampled[:, :, ::-1] # BGR->RGB

    @staticmethod
    def __extract(image, face_location, scale):
        '''Upscale coordinates and extract face'''
        factor = int(1 / scale) if scale > 0 else 1
        top, right, bottom, left = face_location
        return image[top * factor:bottom * factor, left * factor:right * factor]

    @staticmethod
    def __format_name(output_path, name):
        return path.join(output_path, '{}.jpg'.format(name))

    @staticmethod
    def __file_count(directory):
        return len([item for item in listdir(directory) if path.isfile(path.join(directory, item))])

    def __check_reference(self, reference):
        '''Checks the type of reference and looks for encodings'''
        if path.isdir(reference):
            with tqdm(total=self.__file_count(reference), unit='files') as progress:
                for file in listdir(reference):
                    progress.update(1)
                    progress.set_description('Checking reference: {}'.format(file))
                    progress.refresh()
                    self.__parse_encoding(path.join(reference, file))
        elif path.isfile(reference):
            self.__parse_encoding(reference)
        if self.reference_count:
            print('Found {} face references'.format(self.reference_count))
        else:
            print('Warning: no face references have been detected')
            print('Are you sure your reference is correct? {}'.format(reference))
            print('If you process a video *all* detected faces will be extracted')

    def __parse_encoding(self, image_path):
        '''Adds the first face encoding in an image to the reference encodings'''
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image, None, self._rs.jitter)
        if numpy.any(encoding):
            self._reference_encodings.append(encoding[0])

    def __recognise(self, face):
        '''Checks a given face against any known reference encodings.
        If no reference encodings are present any face is classed as recognised.'''
        if not self.reference_count:
            return True
        encoding = face_recognition.face_encodings(face, None, self._rs.jitter)
        if numpy.any(encoding):
            return numpy.any(face_recognition.compare_faces(self._reference_encodings,
                                                            encoding[0],
                                                            self._rs.tolerance))
        return False

    def __reset_frames(self):
        self._process_frames = []
        self._orignal_frames = []

    def __get_face_locations(self):
        '''Get the batch face locations and frame number'''
        batch = face_recognition.batch_face_locations(self._process_frames, 1, self._ps.batch_size)
        for index, locations in enumerate(batch):
            yield (index, locations)

    def __get_faces(self, face_locations, position):
        '''Get the faces from a set of locations'''
        for _, location in enumerate(face_locations):
            face = self.__extract(self._orignal_frames[position], location, self._ps.scale)
            yield face

    def __save_extract(self, face, file_path):
        '''Saves the face to file_path at the set extract size'''
        image = cv2.resize(face, (self._ps.extract_size, self._ps.extract_size))
        cv2.imwrite(file_path, image)
        self._total_extracted += 1

    def __get_fame(self, sequence):
        '''Grabs, decodes and returns the next frame and number.'''
        frame_count = 0
        while sequence.isOpened():
            ret, frame = sequence.read()
            if not ret:
                break
            frame_count += 1
            if self.__skip_frame(frame_count):
                continue
            yield (frame, frame_count)

    def __do_batch(self, batch_count, output_path):
        '''Handles each batch of detected faces, performing recognition on each'''
        with tqdm(total=self._ps.batch_size, unit='frame') as progress:
            extracted = 0
            for location_index, locations in self.__get_face_locations():
                progress.update(1)
                progress.set_description('Batch #{} (recognised {})'.format(batch_count, extracted))
                for face in self.__get_faces(locations, location_index):
                    if self.__recognise(face):
                        extracted += 1
                        name = self.__format_name(output_path, self._total_extracted)
                        self.__save_extract(face, name)
                        # frame v.unlikely to have target face more than once
                        # however this only holds true if we have a reference
                        if self.reference_count:
                            break

    def __skip_frame(self, number):
        '''We want every nth frame if skipping'''
        return self._ps.skip_frames > 0 and number % self._ps.skip_frames

    def __batch_builder(self, output_path, sequence, total_frames):
        '''Splits the fames in batches and keeps score'''
        with tqdm(total=total_frames, unit='frame') as progress:
            batch_count = 0
            for frame, frame_count in self.__get_fame(sequence):
                progress.update(frame_count - progress.n)
                progress.set_description('Total (extracted {})'.format(self._total_extracted))
                self._process_frames.append(self.__downsample(frame, self._ps.scale))
                self._orignal_frames.append(frame)
                if len(self._process_frames) == self._ps.batch_size:
                    batch_count += 1
                    self.__do_batch(batch_count, output_path)
                    self.__reset_frames()

    def process(self, input_path, output_path='.'):
        '''
        Extracts known faces from the input source to the output.
            :param str input_path: Path to video or image sequence pattern
            :param str output_path: path to output directory
        '''
        self._total_extracted = 0
        sequence = cv2.VideoCapture(input_path)
        total_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
        total_work = int(total_frames / self._ps.skip_frames)
        total_batches = int(total_work / self._ps.batch_size)
        total_refs = len(self._reference_encodings)
        print('Processing {} at {} scale'.format(input_path, self._ps.scale))
        print('Using {} reference{} ({} jitter {} tolerance)'.format(total_refs,
                                                                     's' if total_refs > 1 else '',
                                                                     self._rs.jitter,
                                                                     self._rs.tolerance))
        print('Checking {} of {} frames in {} batches of {}'.format(total_work,
                                                                    total_frames,
                                                                    total_batches,
                                                                    self._ps.batch_size))
        self.__batch_builder(output_path, sequence, total_frames)

if __name__ == '__main__':
    AP = argparse.ArgumentParser(description='''FaceGrab''')
    # Required settings
    AP.add_argument('-r', '--reference', type=str, required=True,
                    help=r'''Path to a single file e.g. ./images/someone.jpg
    or a path to a directory of reference images e.g. ./images.
    (You can also pass an empty directory if you wish to match all faces).''')
    AP.add_argument('-i', '--input', type=str, required=True,
                    help=r'''Path to a single file e.g. ./video/foo.mp4
    Or a path/pattern of an image sequence e.g. ./frames/img_%%04d.jpg
    (read like ./frames/img_0000.jpg, ./frames/img_0001.jpg, ./frames/img_0002.jpg, ...)''')
    AP.add_argument('-o', '--output', type=str, required=True,
                    help='''Path to output directory''')
    # Optional process settings
    AP.add_argument('-bs', '--batch_size', type=int, default=128,
                    help='''How many frames to include in each GPU processing batch.''')
    AP.add_argument('-sf', '--skip_frames', type=int, default=6,
                    help='''How many frames to skip e.g. 5 means look at every 6th''')
    AP.add_argument('-xs', '--extract_size', type=int, default=256,
                    help='''Size in pixels of extracted face images (n*n).''')
    AP.add_argument('-s', '--scale', type=int, default=0.25,
                    help='''Factor to down-sample input by for detection processing.
    If you get too few matches try scaling by half e.g. 0.5''')
    # Optional recognition settings
    AP.add_argument('-t', '--tolerance', type=float, default=0.6,
                    help='''How much "distance" between faces to consider it a match.
    Lower is stricter. 0.6 is typical best performance''')
    AP.add_argument('-j', '--jitter', type=int, default=5,
                    help='''How many times to re-sample images when
    calculating recognition encodings. Higher is more accurate, but slower.
    (100 is 100 times slower than 1).''')
    ARGS = AP.parse_args()
    RS = RecognitionSettings(tolerance=ARGS.tolerance, jitter=ARGS.jitter)
    PS = ProcessSettings(batch_size=ARGS.batch_size,
                         skip_frames=ARGS.skip_frames,
                         extract_size=ARGS.extract_size,
                         scale=ARGS.scale)
    FG = FaceGrab(ARGS.reference, RS, PS)
    FG.process(ARGS.input, ARGS.output)
