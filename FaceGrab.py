'''
Extract a known face from a video.

This class uses a combination of a deep learning CNN model to batch detect faces
in video frames, or a sequence of images, in GPU with CUDA.
It then uses HoG to compare the detected faces with a computed reference set of face encodings.
'''

from os import path, listdir
from tqdm import tqdm

import cv2
import numpy
import face_recognition

class FaceGrab():
    r'''
    Common settings for reference encodings and processing parameters.
    so that multiple videos can be processed against them.

    :param str reference: can be a path to a single file e.g. .\images\someone.jpg
                or a path to a directory of images e.g. .\images
    :param int batch_size: How many images to include in each GPU processing batch.
    :param int skip_frames: How many frame to skip e.g. 5 means look at every 6th
    :param float tolerance: How much distance between faces to consider it a match.
                Lower is more strict. 0.6 is typical best performance.
    :param int extract_size: Size in pixels of extracted face images (NxN)
    :param int reference_jitter: How many times to re-sample a face when
                       calculating encoding. Higher is more accurate,
                       but slower (i.e. 100 is 100x slower)
    '''
    def __init__(self,
                 reference,
                 batch_size=128,
                 skip_frames=5,
                 tolerance=0.6,
                 extract_size=256,
                 reference_jitter=100):
        self.batch_size = numpy.clip(batch_size, 2, 128)
        self.skip_frames = 0 if skip_frames < 0 else skip_frames + 1
        self.tolernace = numpy.clip(tolerance, 0.1, 1)
        self.extract_size = extract_size
        self.reference_jitter = reference_jitter
        self._process_frames = []
        self._orignal_frames = []
        self._reference_encodings = []
        self._total_extracted = 0
        self.__check_reference(reference)

    @property
    def has_references(self):
        '''True if any encodings are currently loaded'''
        return numpy.any(self._reference_encodings)

    @staticmethod
    def __downsample(frame, scale):
        '''Downscale frame of video for faster detection processing.
        also converts cv2's BGR to face_recognition's RGB'''
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        small_frame = small_frame[:, :, ::-1] # BGR->RGB
        return small_frame

    @staticmethod
    def __extract(frame, location, scale):
        '''Upscale coordinates and extract face'''
        factor = int(1 / scale)
        top, right, bottom, left = location
        return frame[top * factor:bottom * factor, left * factor:right * factor]

    def __check_reference(self, reference):
        '''Checks the type of reference and looks for encodings'''
        if path.isdir(reference):
            for file in listdir(reference):
                self.__parse_encoding(path.join(reference, file))
        elif path.isfile(reference):
            self.__parse_encoding(reference)
        if not self.has_references:
            print('Warning: no references have been detected')
            print('Are you sure the reference is correct? {}'.format(reference))
            print('If you process a video *all* detected faces will be extracted')

    def __parse_encoding(self, image_path):
        '''Adds the first face encoding in an image to the known encodings'''
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image, None, self.reference_jitter)
        if numpy.any(encoding):
            self._reference_encodings.append(encoding[0])
            print('Found ref #{} in {}'.format(len(self._reference_encodings), image_path))

    def __recognise(self, face):
        '''Checks a face against any known reference encodings.
        If no reference encodings are present *all* faces are classed as recognised.'''
        if not self.has_references:
            return True
        encoding = face_recognition.face_encodings(face)
        if numpy.any(encoding):
            return numpy.any(face_recognition.compare_faces(self._reference_encodings,
                                                            encoding[0],
                                                            self.tolernace))
        return False

    def __reset_frames(self):
        self._process_frames = []
        self._orignal_frames = []

    def __do_batch(self, batch_count, frame_count, output_path, scale):
        '''Detects face in batch of frames and tests to see if each face is recognised.
        If a face is recognised it is saved as an image to the output path'''
        batch = face_recognition.batch_face_locations(self._process_frames, 1, self.batch_size)
        with tqdm(total=len(batch)) as progress:
            extracted = 0
            for position, found in enumerate(batch):
                frame = frame_count - self.batch_size + position
                progress.update(1)
                progress.set_description('Batch #{} (recognised {})'.format(batch_count, extracted))
                for count, location in enumerate(found):
                    face = self.__extract(self._orignal_frames[position], location, scale)
                    if self.__recognise(face):
                        extracted += 1
                        self._total_extracted += 1
                        out = path.join(output_path, '{}-{}-{}.jpg'.format(frame, position, count))
                        cv2.imwrite(out, cv2.resize(face, (self.extract_size, self.extract_size)))
                        # frame v.unlikely to have target face more than once
                        # however this only holds if we have a reference
                        if self.has_references:
                            break

    def __skip_frame(self, number):
        '''We want every nth frame if skipping'''
        return self.skip_frames > 0 and number % self.skip_frames

    def __batch_builder(self, output_path, sequence, total_frames, scale):
        frame_count = 0
        batch_count = 0
        with tqdm(total=total_frames) as progress:
            while sequence.isOpened():
                ret, frame = sequence.read()
                if not ret:
                    break
                frame_count += 1
                progress.update(1)
                progress.set_description('Total (extracted {})'.format(self._total_extracted))
                if self.__skip_frame(frame_count):
                    continue
                self._process_frames.append(self.__downsample(frame, scale))
                self._orignal_frames.append(frame)
                if len(self._process_frames) == self.batch_size:
                    batch_count += 1
                    self.__do_batch(batch_count, frame_count, output_path, scale)
                    self.__reset_frames()

    def process(self, input_path, output_path='.', scale=0.25):
        r'''
        Extracts images from the input to the output.

        :param str input_path: path to a single file e.g. .\video\foo.mp4
                     or a path to an image sequence e.g. .\frames\img_%04d.jpg
                     (read like img_0000.jpg, img_0001.jpg, img_0002.jpg, ...)
        :param str output_path: path to output directory
        :param float scale: Amount to down-sampled input by for detection processing
                            if you get too few matches try scaling by half e.g. .5
        '''
        scale = numpy.clip(scale, 0, 1.0)
        self._total_extracted = 0
        sequence = cv2.VideoCapture(input_path)
        total_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
        total_work = int(total_frames / self.skip_frames)
        total_batches = int(total_work / self.batch_size)
        total_refs = len(self._reference_encodings)
        print('Processing {} at {} scale'.format(input_path, scale))
        print('Using {} reference{} ({} jitter {} tolerance)'.format(total_refs,
                                                                     's' if total_refs > 1 else '',
                                                                     self.reference_jitter,
                                                                     self.tolernace))
        print('Checking {} of {} frames in {} batches of {}'.format(total_work,
                                                                    total_frames,
                                                                    total_batches,
                                                                    self.batch_size))
        self.__batch_builder(output_path, sequence, total_frames, scale)
        sequence.release()

if __name__ == '__main__':
    # NB: If run out of memory either reduce the batch_size
    # roughly speaking
    # batch_size * frame dimensions * process scale = VRAM needed
    FG = FaceGrab(reference=r'D:\ref',
                  batch_size=128,
                  skip_frames=12,
                  tolerance=.6,
                  extract_size=256,
                  reference_jitter=50)
    if FG.has_references:
        FG.process(input_path=r'D:\Videos\Movies\Gladiator (2000)\Gladiator (2000).avi',
                   output_path=r'D:\out',
                   scale=.25)
