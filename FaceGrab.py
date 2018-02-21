'''
Extract a known face from a video.

Uses a combination of a deep learning CNN model to batch detect faces
in video frames, or a sequence of images, in GPU with CUDA and HoG to compare
the detected faces with a computed reference set of face encodings.
'''
from os import path, listdir
from typing import NamedTuple

import cv2
import numpy as np
import face_recognition as fr
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
        :param bool display_output: Show the detection and extraction images in process.
    '''
    batch_size: int = 128
    skip_frames: int = 6
    extract_size: int = 256
    scale: float = .25
    display_output: bool = False

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
        self.__ps = process._replace(batch_size=np.clip(process.batch_size, 2, 128),
                                     skip_frames=skip_sanity,
                                     scale=np.clip(process.scale, 0, 1.0))
        self.__rs = recognition._replace(tolerance=np.clip(recognition.tolerance, 0.1, 1))
        self.__process_frames = []
        self.__original_frames = []
        self.__reference_encodings = []
        self.__total_extracted = 0
        self.__check_reference(reference)

    _MEAN_FACE_LANDMARKS = np.asarray([
        [2.13256e-04, 1.06454e-01], [7.52622e-02, 3.89150e-02], [1.81130e-01, 1.87482e-02],
        [2.90770e-01, 3.44891e-02], [3.93397e-01, 7.73906e-02], [5.86856e-01, 7.73906e-02],
        [6.89483e-01, 3.44891e-02], [7.99124e-01, 1.87482e-02], [9.04991e-01, 3.89150e-02],
        [9.80040e-01, 1.06454e-01], [4.90127e-01, 2.03352e-01], [4.90127e-01, 3.07009e-01],
        [4.90127e-01, 4.09805e-01], [4.90127e-01, 5.15625e-01], [3.66880e-01, 5.87326e-01],
        [4.26036e-01, 6.09345e-01], [4.90127e-01, 6.28106e-01], [5.54217e-01, 6.09345e-01],
        [6.13373e-01, 5.87326e-01], [1.21737e-01, 2.16423e-01], [1.87122e-01, 1.78758e-01],
        [2.65825e-01, 1.79852e-01], [3.34606e-01, 2.31733e-01], [2.60918e-01, 2.45099e-01],
        [1.82743e-01, 2.44077e-01], [6.45647e-01, 2.31733e-01], [7.14428e-01, 1.79852e-01],
        [7.93132e-01, 1.78758e-01], [8.58516e-01, 2.16423e-01], [7.97510e-01, 2.44077e-01],
        [7.19335e-01, 2.45099e-01], [2.54149e-01, 7.80233e-01], [3.40985e-01, 7.45405e-01],
        [4.28858e-01, 7.27388e-01], [4.90127e-01, 7.42578e-01], [5.51395e-01, 7.27388e-01],
        [6.39268e-01, 7.45405e-01], [7.26104e-01, 7.80233e-01], [6.42159e-01, 8.64805e-01],
        [5.56721e-01, 9.02192e-01], [4.90127e-01, 9.09281e-01], [4.23532e-01, 9.02192e-01],
        [3.38094e-01, 8.64805e-01], [2.90379e-01, 7.84792e-01], [4.28096e-01, 7.78746e-01],
        [4.90127e-01, 7.85343e-01], [5.52157e-01, 7.78746e-01], [6.89874e-01, 7.84792e-01],
        [5.53364e-01, 8.24182e-01], [4.90127e-01, 8.31803e-01], [4.26890e-01, 8.24182e-01]])

    @property
    def reference_count(self):
        '''Total currently loaded reference encodings for recognition'''
        return len(self.__reference_encodings)

    @staticmethod
    def __downsample(image, scale):
        '''Downscale and convert image for faster detection processing'''
        sampled = cv2.resize(image,
                             (0, 0),
                             fx=scale,
                             fy=scale,
                             interpolation=cv2.INTER_AREA) if scale > 0 else image
        return sampled[:, :, ::-1] #RGB->BGR

    @classmethod
    def __extract(cls, image, location, scale, size):
        '''Upscale coordinates and extract location from image at the given size'''
        factor = int(1 / scale) if scale > 0 else 1
        location_scaled = tuple(factor * n for n in location)
        top, right, bottom, left = location_scaled
        # protected member access - eek!
        landmarks = fr.api._raw_face_landmarks(image, [location_scaled])
        if not any(landmarks):
            print('Warning landmarks not detected - falling back to crop')
            return cv2.resize(image[top:bottom, left:right],
                              (size, size),
                              interpolation=cv2.INTER_AREA)
        # TODO : would be much faster to always crop for recognition
        # and transform on save - however recognition rate against
        # untransformed faces is much lower...
        return cls.__transform(image, landmarks[0], size, 48)

    @classmethod
    def __transform(cls, image, landmarks, size, padding=0):
        '''Warps the face based on a hard-coded mean face value matrix'''
        coordinates = [(p.x, p.y) for p in landmarks.parts()]
        mat = cls.__umeyama(np.asarray(coordinates[17:]))[0:2]
        mat = mat * (size - 2 * padding)
        mat[:, 2] += padding
        return cv2.warpAffine(image, mat, (size, size))

    @classmethod
    def __umeyama(cls, face):
        '''
        N-D similarity transform with scaling.
        Adapted from:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py
        http://web.stanford.edu/class/cs273/refs/umeyama.pdf
        '''
        N, m = face.shape
        mx = face.mean(axis=0)
        my = np.average(cls._MEAN_FACE_LANDMARKS, 0)
        dx = face - mx
        dy = cls._MEAN_FACE_LANDMARKS - my #hard-code T?
        A = np.dot(dy.T, dx) / N
        d = np.ones((m,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[m - 1] = -1
        T = np.eye(m + 1, dtype=np.double)
        U, S, V = np.linalg.svd(A)
        rank = np.linalg.matrix_rank(A) #covariance
        if rank == 0:
            return np.nan * T
        elif rank == m - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:m, :m] = np.dot(U, V)
            else:
                s = d[m - 1]
                d[m - 1] = -1
                T[:m, :m] = np.dot(U, np.dot(np.diag(d), V))
                d[m - 1] = s
        else:
            T[:m, :m] = np.dot(U, np.dot(np.diag(d), V))
        scale = 1.0 / dx.var(axis=0).sum() * np.dot(S, d)
        T[:m, m] = my - scale * np.dot(T[:m, :m], mx.T)
        T[:m, :m] *= scale
        return T

    @staticmethod
    def __file_count(directory):
        '''Returns the number of files in a directory'''
        return len([item for item in listdir(directory) if path.isfile(path.join(directory, item))])

    @staticmethod
    def __draw_detection(frame, locations):
        '''draws the process frames with face detection locations'''
        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 1)
        cv2.imshow('process', frame[:, :, ::-1]) #BGR->RGB
        cv2.waitKey(delay=1)

    def __check_reference(self, reference):
        '''Checks if the reference is a wild-card/directory/file and looks for encodings'''
        if reference == '*':
            return
        if path.isdir(reference):
            with tqdm(total=self.__file_count(reference), unit='files') as progress:
                for file in listdir(reference):
                    full_path = path.join(reference, file)
                    if path.isfile(full_path):
                        progress.update(1)
                        progress.set_description(f'Checking reference: {file}')
                        self.__parse_encoding(full_path)
            return
        if path.isfile(reference):
            if reference.endswith('.npz'): #saved
                self.load(reference)
                return
            self.__parse_encoding(reference)
            return
        raise ValueError(f'Invalid reference: {reference}')

    def __parse_encoding(self, image_path):
        '''Adds the first face encoding in an image to the reference encodings'''
        image = fr.load_image_file(image_path)
        encoding = fr.face_encodings(image, None, self.__rs.jitter)
        if np.any(encoding):
            self.__reference_encodings.append(encoding[0])

    def __recognise(self, face):
        '''Checks a given face against any known reference encodings.
        If face is within the tolerance of any reference it is classed as recognised.
        If no reference encodings are present any face is classed as recognised.'''
        if not self.reference_count:
            return True
        encoding = fr.face_encodings(face, None, self.__rs.jitter)
        if np.any(encoding):
            return np.any(fr.compare_faces(self.__reference_encodings,
                                           encoding[0],
                                           self.__rs.tolerance))
        return False

    def __reset_frames(self):
        self.__process_frames.clear()
        self.__original_frames.clear()

    def __get_location_frames(self):
        '''Get the total faces detected and f.indexed locations/original/process based on hits.'''
        batch = fr.batch_face_locations(self.__process_frames, 1, self.__ps.batch_size)
        hits = np.nonzero(batch)[0] # fancy
        locations = np.asarray(batch)[hits]
        original = np.asarray(self.__original_frames)[hits]
        process = np.asarray(self.__process_frames)[hits]
        total = sum(len(x) for x in locations)

        return (total, zip(locations, original, process))

    def __get_faces(self, image, face_locations):
        '''Get the faces from a set of locations'''
        for location in face_locations:
            face = self.__extract(image, location, self.__ps.scale, self.__ps.extract_size)
            yield face

    def __save_extract(self, image, file_path):
        '''Saves the face to file_path at the set extract size'''
        cv2.imwrite(file_path, image)
        self.__total_extracted += 1
        if self.__ps.display_output:
            cv2.imshow('extracted', image)
            cv2.waitKey(delay=1)

    def __get_fame(self, sequence, total_frames):
        '''Grabs, decodes and returns the next frame and number.'''
        for frame_number in range(total_frames):
            if self.__skip_frame(frame_number): # skip *then* read
                continue
            sequence.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = sequence.read()
            if not ret:
                break
            yield (frame, frame_number)

    def __do_batch(self, batch_count, output_path):
        '''Handles each batch of detected faces, performing recognition on each'''
        total, results = self.__get_location_frames()
        self.__reset_frames()
        with tqdm(total=total, unit='checks') as progress:
            progress.set_description(f'Batch #{batch_count}')
            if not total:
                return
            extracted = 0
            # each set of face locations in the batch
            for locations, original, processed in results:
                if self.__ps.display_output:
                    self.__draw_detection(processed, locations)
                for face in self.__get_faces(original, locations):
                    progress.update(1)
                    progress.set_description(f'Batch #{batch_count} (recognised {extracted})')
                    if self.__recognise(face):
                        extracted += 1
                        name = path.join(output_path, f'{self.__total_extracted}.jpg')
                        self.__save_extract(face, name)
                        # image v.unlikely to have target face more than once
                        # however this only holds true if we have a reference
                        if self.reference_count:
                            break

    def __skip_frame(self, number):
        '''We want every nth frame if skipping'''
        return self.__ps.skip_frames > 0 and number % self.__ps.skip_frames

    def __batch_builder(self, output_path, sequence, total_frames):
        '''Splits the fames in batches and keeps score'''
        with tqdm(total=total_frames, unit='frame') as progress:
            batch_count = 0
            for frame, frame_count in self.__get_fame(sequence, total_frames):
                progress.update(frame_count - progress.n)
                progress.set_description(f'Total (extracted {self.__total_extracted})')
                process_frame = self.__downsample(frame, self.__ps.scale)
                self.__process_frames.append(process_frame)
                self.__original_frames.append(frame)
                if self.__ps.display_output:
                    self.__draw_detection(process_frame, [])
                if len(self.__process_frames) == self.__ps.batch_size:
                    batch_count += 1
                    self.__do_batch(batch_count, output_path)

    def save(self, file_path):
        '''
        Saves references in compressed npz format to the given path
            :param str file_path: Path to save file (.npz extension will be added if not present)
        '''
        print(f'Saving {len(self.__reference_encodings)} to {file_path}.npz')
        np.savez_compressed(file_path, *self.__reference_encodings)

    def load(self, file_path):
        '''
        Loads references in compressed npz format from the given path.
        NB: Overwrites any existing encodings
            :param str file_path: Path to a .npz file generated from the save method
        '''
        npzfile = np.load(file_path)
        print(f'Loading {len(npzfile.files)} from {file_path}')
        self.__reference_encodings = [npzfile[key] for key in npzfile]

    def process(self, input_path, output_path='.'):
        '''
        Extracts known faces from the input source to the output.
            :param str input_path: Path to video or image sequence pattern
            :param str output_path: path to output directory
        '''
        self.__total_extracted = 0
        sequence = cv2.VideoCapture(input_path)
        frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
        work = int(frames / self.__ps.skip_frames)
        batches = int(work / self.__ps.batch_size)
        print(f'Processing {input_path} ({self.__ps.scale} scale)')
        print(f'References {self.reference_count} at {self.__rs.tolerance} tolerance')
        print(f'Checking {work} of {frames} frames in {batches} batches of {self.__ps.batch_size}')
        # prevent v.intermittent
        # 'tqdm' object has no attribute 'miniters'
        tqdm.monitor_interval = 0
        self.__batch_builder(output_path, sequence, frames)

if __name__ == '__main__':
    import argparse

    class Range(object):
        '''Restricted range for float arguments'''
        def __init__(self, start, end):
            self.start = start
            self.end = end
        def __eq__(self, other):
            return self.start <= other <= self.end

    AP = argparse.ArgumentParser(description='''FaceGrab''')
    # Required settings
    AP.add_argument('-r', '--reference', type=str, required=True,
                    help=r'''Path to a single image file or a directory of reference images.
                    Also accepts the path to an .npz file generated by save_references.
                    Finally to skip recognition and extract all faces an asterix '*'
                    can be passed.''')
    AP.add_argument('-i', '--input', type=str, required=True,
                    help=r'''Path to a single file e.g. ./video/foo.mp4
    Or a path/pattern of an image sequence e.g. ./frames/img_%%04d.jpg
    (read like ./frames/img_0000.jpg, ./frames/img_0001.jpg, ./frames/img_0002.jpg, ...)''')
    AP.add_argument('-o', '--output', type=str, required=True,
                    help='''Path to output directory''')
    # Optional save settings
    AP.add_argument('-sr', '--save_references', type=str,
                    help='''Save the references in .npz format.
                    This file can be loaded as a reference avoiding the need
                    to re-encode a set of images each time for the same person''')
    # Optional process settings
    AP.add_argument('-bs', '--batch_size', type=int, default=128, choices=range(2, 128),
                    metavar='[2-128]',
                    help='''How many frames to include in each GPU processing batch.''')
    AP.add_argument('-sf', '--skip_frames', type=int, default=6, choices=range(0, 1000),
                    metavar='[0-1000]',
                    help='''How many frames to skip e.g. 5 means look at every 6th''')
    AP.add_argument('-xs', '--extract_size', type=int, default=256, choices=range(32, 1024),
                    metavar='[32-1024]',
                    help='''Size in pixels of extracted face images (n*n).''')
    AP.add_argument('-s', '--scale', type=float, default=0.25, choices=[Range(0.1, 1.0)],
                    metavar='[0.1-1.0]',
                    help='''Factor to down-sample input by for detection processing.
    If you get too few matches try scaling by half e.g. 0.5''')
    AP.add_argument('-do', '--display_output', action='store_true',
                    help='''Show the detection and extraction images (slows processing).''')
    # Optional recognition settings
    AP.add_argument('-t', '--tolerance', type=float, default=0.6, choices=[Range(0.1, 1.0)],
                    metavar='[0.1-1.0]',
                    help='''How much "distance" between faces to consider it a match.
    Lower is stricter. 0.6 is typical best performance''')
    AP.add_argument('-j', '--jitter', type=int, default=5, choices=range(1, 1000),
                    metavar='[1-1000]',
                    help='''How many times to re-sample images when
    calculating recognition encodings. Higher is more accurate, but slower.
    (100 is 100 times slower than 1).''')
    ARGS = AP.parse_args()
    RS = RecognitionSettings(tolerance=ARGS.tolerance, jitter=ARGS.jitter)
    PS = ProcessSettings(batch_size=ARGS.batch_size,
                         skip_frames=ARGS.skip_frames,
                         extract_size=ARGS.extract_size,
                         scale=ARGS.scale,
                         display_output=ARGS.display_output)
    FG = FaceGrab(ARGS.reference, RS, PS)
    if ARGS.save_references:
        FG.save(ARGS.save_references)
    FG.process(ARGS.input, ARGS.output)
