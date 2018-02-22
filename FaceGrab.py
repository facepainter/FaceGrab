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
    jitter: int = 5

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
        self.__extract_dim = (self.__ps.extract_size, self.__ps.extract_size)
        # TODO: prop/arg @18%256=46 
        self.__extract_pad = int(self.__ps.extract_size / 100 * 18) 
        self.__check_reference(reference)

    _MEAN_FACE_TRANSPOSED = np.asarray([
        [-0.4899134, -0.41486446, -0.30899666, -0.19935666, -0.09672966,
         0.09672934, 0.19935634, 0.30899734, 0.41486434, 0.48991334,
         0.00000034, 0.00000034, 0.00000034, 0.00000034, -0.12324666,
         -0.06409066, 0.00000034, 0.06409034, 0.12324634, -0.36838966,
         -0.30300466, -0.22430166, -0.15552066, -0.22920866, -0.30738366,
         0.15552034, 0.22430134, 0.30300534, 0.36838934, 0.30738334,
         0.22920834, -0.23597766, -0.14914166, -0.06126866, 0.00000034,
         0.06126834, 0.14914134, 0.23597734, 0.15203234, 0.06659434,
         0.00000034, -0.06659466, -0.15203266, -0.19974766, -0.06203066,
         0.00000034, 0.06203034, 0.19974734, 0.06323734, 0.00000034,
         -0.06323666],
        [-0.35796968, -0.42550868, -0.44567548, -0.42993458, -0.38703308,
         -0.38703308, -0.42993458, -0.44567548, -0.42550868, -0.35796968,
         -0.26107168, -0.15741468, -0.05461868, 0.05120132, 0.12290232,
         0.14492132, 0.16368232, 0.14492132, 0.12290232, -0.24800068,
         -0.28566568, -0.28457168, -0.23269068, -0.21932468, -0.22034668,
         -0.23269068, -0.28457168, -0.28566568, -0.24800068, -0.22034668,
         -0.21932468, 0.31580932, 0.28098132, 0.26296432, 0.27815432,
         0.26296432, 0.28098132, 0.31580932, 0.40038132, 0.43776832,
         0.44485732, 0.43776832, 0.40038132, 0.32036832, 0.31432232,
         0.32091932, 0.31432232, 0.32036832, 0.35975832, 0.36737932,
         0.35975832]])

    @classmethod
    def __get_matrix(cls, face):
        '''
        Estimated N-D similarity transform with scaling adapted from _umeyama
        https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py#L72
        http://web.stanford.edu/class/cs273/refs/umeyama.pdf
        '''
        mean_face = np.average(face, axis=0)
        mean_reference = np.asarray([0.49012666, 0.46442368])
        dx = face - mean_face
        d = np.ones((2,))
        T = np.identity(3)
        U, s, V = np.linalg.svd(np.dot(cls._MEAN_FACE_TRANSPOSED, dx) / 51)
        T[:2, :2] = np.dot(U, np.dot(np.diag(d), V))
        scale = 1.0 / dx.var(axis=0).sum() * np.dot(s, d)
        T[:2, 2] = mean_reference - scale * np.dot(T[:2, :2], mean_face.T)
        T[:2, :2] *= scale
        return T

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

    def __transform(self, image, landmarks, padding=0):
        '''Affine transform between image landmarks and "mean face"'''
        coordinates = [(p.x, p.y) for p in landmarks.parts()]
        mat = self.__get_matrix(np.asarray(coordinates[17:]))[0:2]
        mat = mat * (self.__ps.extract_size - 2 * padding)
        mat[:, 2] += padding
        return cv2.warpAffine(image, mat, self.__extract_dim, None, flags=cv2.INTER_LINEAR)

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
            if reference.endswith('.npz'):
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
        else:
            tqdm.write(f'No encoding: {image_path}')

    def __recognise(self, face):
        '''Checks a given face against any known reference encodings.
        Returns total number of reference encodings below or equal to tolerance.
        Recognised returns a number between 1 and total number of reference encodings.
        NOT recognised returns 0
        NO reference encodings are loaded returns -1
        NO face encoding can be found returns 0'''
        if not self.reference_count:
            return -1
        encoding = fr.face_encodings(face, None, self.__rs.jitter)
        if not np.any(encoding):
            return 0
        # maybe? np.mean(distance[np.where(distance <= self.__rs.tolerance)[0]])
        distance = fr.face_distance(self.__reference_encodings, encoding[0])
        return len(np.where(distance <= self.__rs.tolerance)[0])

    def __reset_frames(self):
        self.__process_frames.clear()
        self.__original_frames.clear()

    def __get_location_frames(self):
        '''Get the total faces detected and f.indexed locations/original/process based on hits.'''
        batch = fr.batch_face_locations(self.__process_frames, 1, self.__ps.batch_size)
        hits = np.nonzero(batch)[0]
        locations = np.asarray(batch)[hits]
        return (sum(len(x) for x in locations),
                zip(locations,
                    np.asarray(self.__original_frames)[hits],
                    np.asarray(self.__process_frames)[hits]))

    def __get_faces(self, image, face_locations):
        '''Get the quick cropped faces and scaled locations from
        a set of detect locations'''
        for location in face_locations:
            factor = int(1 / self.__ps.scale)
            scaled = tuple(factor * n for n in location)
            top, right, bottom, left = scaled
            yield (image[top:bottom, left:right], scaled)

    def __save_extract(self, image, file_path):
        '''Saves the image to file_path at the extract dimensions'''
        width, height, _ = np.shape(image)
        if (width, height) != self.__extract_dim:
            image = cv2.resize(image, self.__extract_dim, interpolation=cv2.INTER_AREA)
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
        extracted = 0
        with tqdm(total=total, unit='checks') as progress:
            progress.set_description(f'Batch #{batch_count}')
            if not total:
                return
            for locations, original, processed in results:
                if self.__ps.display_output:
                    self.__draw_detection(processed, locations)
                for face, location_scaled in self.__get_faces(original, locations):
                    progress.update(1)
                    progress.set_description(f'Batch #{batch_count} (recognised {extracted})')
                    recognised = self.__recognise(face)
                    if recognised:
                        extracted += 1
                        # NB: protected-access - eek!
                        landmarks = fr.api._raw_face_landmarks(original, [location_scaled])
                        if any(landmarks):
                            face = self.__transform(original, landmarks[0], self.__extract_pad)
                        name = path.join(output_path, f'{recognised}-{self.__total_extracted}.jpg')
                        self.__save_extract(face, name)
                        # v.unlikely to have target multiple times
                        # break if we have references
                        if self.reference_count:
                            break

    def __skip_frame(self, number):
        '''We want every nth frame if skipping'''
        return self.__ps.skip_frames > 0 and number % self.__ps.skip_frames

    def __batch_builder(self, output_path, sequence, total_frames):
        '''Splits the fames in batches and keeps score'''
        tqdm.monitor_interval = 0
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

    def stats(self):
        '''Prints out statistics on the current references'''
        if not self.reference_count:
            print(f'Error: No reference encodings currently loaded')
            return
        matches = []
        distances = []
        for encoding in self.__reference_encodings:
            group = [x for x in self.__reference_encodings if not np.array_equal(x, encoding)]
            if group:
                matches.append(fr.compare_faces(group, encoding, self.__rs.tolerance))
                distances.append(fr.face_distance(group, encoding))
        print(f'Consistancy {round(np.average(matches) * 100, 2)}%')
        print(f'Dist. Avg. {round(np.average(distances), 3)}')
        print(f'Dist. SD. {round(np.std(distances), 3)}')

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
        frame_count = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
        work = int(frame_count / self.__ps.skip_frames)
        batches = int(work / self.__ps.batch_size)
        print(f'Processing {input_path} (scale:{self.__ps.scale})')
        print(f'References {self.reference_count} (tolerance:{self.__rs.tolerance})')
        print(f'Checking {work}/{frame_count} frames in {batches} batches of {self.__ps.batch_size}')
        self.__batch_builder(output_path, sequence, frame_count)

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
    # Optional settings
    AP.add_argument('-ds', '--display_stats', action='store_true',
                    help='''Show the reference encoding statistics (model consistency).''')
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
    if ARGS.display_stats:
        FG.stats()
    if ARGS.save_references:
        FG.save(ARGS.save_references)
    FG.process(ARGS.input, ARGS.output)
