# FaceGrab

Extract aligned images of a known face from a video or image sequence.

Uses a combination of a (precomputed) deep learning CNN model to quickly batch detect faces
in video frames then HoG face recognition with single or multiple encodings computed from known references

Using the GPU with CUDA in this way means batch processing face detection in up to 128 frames
at at time can be achieved (VRAM dependant). This combined with other speed/optimisation techniques
(such as downsampling the process frames, early frame skipping, etc) means that very high quality, low false positive, 
faces can be extracted from video or image sequences file many times faster than using seperate
frame splitting/extraction/detection applications, or by methods that are CPU bound and only operate on individual images.

> One important caveat in this process is that the input frames/images must be exactly the same dimensions.
> As this is primarily geared towards extraction from video this should not be an issue.
> However it is worth bearing in mind should you get errors processing image sequences :)

## Usage

### Script based

Example

```
python facegrab.py -r ./pics/russell-crowe -i "./movies/Gladiator (2000).avi" -o ./output 
```

If you like you can also watch the process in action with display_output :)

```
python facegrab.py -r ./pics/russell-crowe -i "./movies/Gladiator (2000).avi" -o ./output --display_output
```

You can save encoding sets used...

```
python facegrab.py -r ./pics/A -i ./video/A.mp4 -o ./extract/A --save_references A.npz
```

And load from the saved file (so that you don't need to save reference images or reprocess for the same person) 

```
python facegrab.py -r A.npz -i ./video/A2.mp4 -o ./extract/A
```

You can get help by passing -h or --help ... you should always ask for help or rtfm :)

```
usage: facegrab.py [-h] -r REFERENCE -i INPUT -o OUTPUT [-sr SAVE_REFERENCES]
                   [-bs [2-128]] [-sf [0-1000]] [-xs [32-1024]] [-s [0.1-1.0]]
                   [-do] [-t [0.1-1.0]] [-j [1-1000]]

FaceGrab

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
                        Path to a single image file or a directory of
                        reference images. Also accepts the path to an .npz
                        file generated by save_references. Finally to skip
                        recognition and extract all faces an asterix '*' can
                        be passed.
  -i INPUT, --input INPUT
                        Path to a single file e.g. ./video/foo.mp4 Or a
                        path/pattern of an image sequence e.g.
                        ./frames/img_%04d.jpg (read like
                        ./frames/img_0000.jpg, ./frames/img_0001.jpg,
                        ./frames/img_0002.jpg, ...)
  -o OUTPUT, --output OUTPUT
                        Path to output directory
  -sr SAVE_REFERENCES, --save_references SAVE_REFERENCES
                        Save the references in npz format. This file can be
                        loaded as a reference avoiding the need to re-encode a
                        set of images each time for the same person
  -bs [2-128], --batch_size [2-128]
                        How many frames to include in each GPU processing
                        batch.
  -sf [0-1000], --skip_frames [0-1000]
                        How many frames to skip e.g. 5 means look at every 6th
  -xs [32-1024], --extract_size [32-1024]
                        Size in pixels of extracted face images (n*n).
  -s [0.1-1.0], --scale [0.1-1.0]
                        Factor to down-sample input by for detection
                        processing. If you get too few matches try scaling by
                        half e.g. 0.5
  -do, --display_output
                        Show the detection and extraction images (slows
                        processing).
  -t [0.1-1.0], --tolerance [0.1-1.0]
                        How much "distance" between faces to consider it a
                        match. Lower is stricter. 0.6 is typical best
                        performance
  -j [1-1000], --jitter [1-1000]
                        How many times to re-sample images when calculating
                        recognition encodings. Higher is more accurate, but
                        slower. (100 is 100 times slower than 1).
```

### Class based

```python
FG = FaceGrab('./images/nick-cage-reference')
FG.process('./movies/The Wicker Man.mp4', './extracted/nick-cage-wicker-man')
```

Or use the Process/Recognition settings to tweak :) 
you can set/miss any or else leave them out entirely 
```python
RS = RecognitionSettings(jitter=1)
PS = ProcessSettings(batch_size=64, extract_size=512, scale=.5)
personA = FaceGrab("someone", RS, PS)
personA.process('a1.mp4', 'a')
personA.process('a2.mp4', 'a')
```

Or like...
```python
personB = FaceGrab("someone-else", process=ProcessSettings(scale=.125))
personB.process('b.mp4', 'b')
personC = FaceGrab("another-person", recognition=RecognitionSettings(tolerance=.4))
personC.process('c.mp4', 'c')
```

Also If you want to ensure you have recognition encodings before you begin...
```python
FG = FaceGrab('./images/nick-cage-reference')
if FG.reference_count:
    FG.process('./movies/The Wicker Man.mp4', './extracted/nick-cage-wicker-man')
```
## Help!

Stuff that might happen that isn't what you wanted or expected...oh cruel world!

### OOM :( - Memory issues 

Very roughly speaking `process.batch_size * [input frame dimensions] * process.scale = VRAM`
As long as you have 2GB+ VRAM and you play with the settings you *should* be golden :)

The two key things being

1. Reduce the **process.batch_size**  - note the whole thing will take longer!
2. Decrease the **process.scale**  e.g. 0.125 (1/8) - you may well get fewer face detections

You could also try re-encoding the video to a lower resolution, but that is cheating and punishable by...nothing.

### If you are getting too many false positives (extracted images of the wrong face/not faces)

1. Use a more varied, higher quality, more representative, range of **reference**  images (ideally ones that look like the person in the input)
2. Increase the **recognition.jitter** so that each encoding/check is done using a higher number of resamples - note this will increase the processing time.
3. Decrease the **recognition.tolerance** so that each recognition is stricter e.g. 0.4

### If you are getting too few matches (missing lots of good images from input)

1. Use a more varied, higher quality, more representative, range of **reference**  images (ideally ones that look like the person in the input)
2. Increase the **recognition.tolerance** so that each recognition is less strict e.g. 0.8
3. Decrease the **recognition.jitter** so that each recognition is done fewer resamples (less accurate) 
4. Decrease the **process.skip_frames** so that more of the input is processed (this might result in very similar extracted images)
5. Increase the **process.scale** e.g. 0.5 (1/2) - bearing in mind you may need to reduce the batch_size accordingly

## Built using
- CUDA 8.0 - https://developer.nvidia.com/cuda-80-ga2-download-archive
- cudnn 6 for cuda 8 - https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170427/cudnn-8.0-windows10-x64-v6.0-zip (login required)
- dlib - https://github.com/davisking/dlib.git compiled with CUDA (and preferably AVX) see notes.
- Visual C++ 2015 Build Tools - http://landinghub.visualstudio.com/visual-cpp-build-tools

YMMV - pretty sure it would work just as well with CUDA 9 / cuDNN 7 / etc - but personally I could not get dlib to build with CUDA support against v9/9.1 :(



