'''
Extract a known face from a video.

Uses a combination of a deep learning CNN model to batch detect faces
in video frames, or a sequence of images, in GPU with CUDA and HoG to compare
the detected faces with a computed reference set of face encodings.
'''
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
        self.__ps = process._replace(batch_size=numpy.clip(process.batch_size, 2, 128),
                                     skip_frames=skip_sanity,
                                     scale=numpy.clip(process.scale, 0, 1.0))
        self.__rs = recognition._replace(tolerance=numpy.clip(recognition.tolerance, 0.1, 1))
        self.__process_frames = []
        self.__original_frames = []
        self.__reference_encodings = []
        self.__total_extracted = 0
        self.__check_reference(reference)

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
        return sampled[:, :, ::-1] # BGR->RGB

    @staticmethod
    def __extract(image, face_location, scale):
        '''Upscale coordinates and extract face'''
        factor = int(1 / scale) if scale > 0 else 1
        top, right, bottom, left = face_location
        return image[top * factor:bottom * factor, left * factor:right * factor]

    @staticmethod
    def __format_name(output_path, name):
        return path.join(output_path, f'{name}.jpg')

    @staticmethod
    def __file_count(directory):
        '''Returns the number of files in a directory'''
        return len([item for item in listdir(directory) if path.isfile(path.join(directory, item))])

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
            self.__parse_encoding(reference)
            return
        raise ValueError(f'Invalid reference: {reference}')

    def __parse_encoding(self, image_path):
        '''Adds the first face encoding in an image to the reference encodings'''
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image, None, self.__rs.jitter)
        if numpy.any(encoding):
            self.__reference_encodings.append(encoding[0])

    def __recognise(self, face):
        '''Checks a given face against any known reference encodings.
        If face is within the tolerance it is classed as recognised.
        If no reference encodings are present any face is classed as recognised.'''
        if not self.reference_count:
            return True
        encoding = face_recognition.face_encodings(face, None, self.__rs.jitter)
        if numpy.any(encoding):
            return numpy.any(face_recognition.compare_faces(self.__reference_encodings,
                                                            encoding[0],
                                                            self.__rs.tolerance))
        return False

    def __reset_frames(self):
        self.__process_frames.clear()
        self.__original_frames.clear()


    def __get_face_locations(self):
        '''Get the total detected and zipped frame numbers/locations'''
        batch = face_recognition.batch_face_locations(self.__process_frames,
                                                      1,
                                                      self.__ps.batch_size)
        hits = numpy.nonzero(batch)[0]
        return (len(hits), zip(hits, numpy.asarray(batch)[hits]))

    def __get_faces(self, image, face_locations):
        '''Get the faces from a set of locations'''
        for _, location in enumerate(face_locations):
            face = self.__extract(image, location, self.__ps.scale)
            yield face

    def __save_extract(self, face, file_path):
        '''Saves the face to file_path at the set extract size'''
        image = cv2.resize(face,
                           (self.__ps.extract_size, self.__ps.extract_size),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite(file_path, image)
        self.__total_extracted += 1
        if self.__ps.display_output:
            cv2.imshow('extracted', image)
            cv2.waitKey(delay=1)

    def __get_fame(self, sequence, total_frames):
        '''Grabs, decodes and returns the next frame and number.'''
        # skip *then* read
        for frame_number in range(total_frames):
            if self.__skip_frame(frame_number):
                continue
            sequence.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = sequence.read()
            if not ret:
                break
            yield (frame, frame_number)

    def __draw_detection(self, idx, locations):
        '''draws the process frame and the face locations
        scaled back on to the original source frames'''
        frame = self.__process_frames[idx][:, :, ::-1] # BGR->RGB
        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 1)
        cv2.imshow('process', frame)
        cv2.waitKey(delay=1)

    def __do_batch(self, batch_count, output_path):
        '''Handles each batch of detected faces, performing recognition on each'''
        total, results = self.__get_face_locations()
        if not total:
            return
        with tqdm(total=total, unit='checks') as progress:
            extracted = 0
            # each set of face locations in the batch
            for idx, locations in results:
                progress.update(1)
                progress.set_description(f'Batch #{batch_count} (recognised {extracted})')
                # display output...
                if self.__ps.display_output:
                    self.__draw_detection(idx, locations)
                # NB: recognition on original image
                for face in self.__get_faces(self.__original_frames[idx], locations):
                    if self.__recognise(face):
                        extracted += 1
                        name = self.__format_name(output_path, self.__total_extracted)
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
                self.__process_frames.append(self.__downsample(frame, self.__ps.scale))
                self.__original_frames.append(frame)
                if len(self.__process_frames) == self.__ps.batch_size:
                    batch_count += 1
                    self.__do_batch(batch_count, output_path)
                    self.__reset_frames()

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
        print(f'References {self.reference_count} at {self.__rs.tolerance} tolerance)')
        print(f'Checking {work} of {frames} frames in {batches} batches of {self.__ps.batch_size}')
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
    AP.add_argument('-bs', '--batch_size', type=int, default=128, choices=range(2, 128),
                    metavar="[2-128]",
                    help='''How many frames to include in each GPU processing batch.''')
    AP.add_argument('-sf', '--skip_frames', type=int, default=6, choices=range(0, 1000),
                    metavar="[0-1000]",
                    help='''How many frames to skip e.g. 5 means look at every 6th''')
    AP.add_argument('-xs', '--extract_size', type=int, default=256, choices=range(32, 1024),
                    metavar="[32-1024]",
                    help='''Size in pixels of extracted face images (n*n).''')
    AP.add_argument('-s', '--scale', type=float, default=0.25, choices=[Range(0.1, 1.0)],
                    metavar="[0.1-1.0]",
                    help='''Factor to down-sample input by for detection processing.
    If you get too few matches try scaling by half e.g. 0.5''')
    AP.add_argument('-do', '--display_output', action='store_true',
                    help='''Show the detection and extraction images (slows processing).''')
    # Optional recognition settings
    AP.add_argument('-t', '--tolerance', type=float, default=0.6, choices=[Range(0.1, 1.0)],
                    metavar="[0.1-1.0]",
                    help='''How much "distance" between faces to consider it a match.
    Lower is stricter. 0.6 is typical best performance''')
    AP.add_argument('-j', '--jitter', type=int, default=5, choices=range(1, 1000),
                    metavar="[1-1000]",
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
    FG.process(ARGS.input, ARGS.output)
