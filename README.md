# FaceGrab

Extract a known face from a video or image sequence.

Uses a combination of a (precomputed) deep learning CNN model to quickly batch detect faces
in video frames then HoG face recognition with a computed reference face encoding or set of encodings.

Using the GPU with CUDA in this way means batch processing face detection in up to 128 frames
at at time can be achieved (VRAM dependant). This combined with other speed/optimisation techniques
(such as downsampling, default frame skipping, etc) means that very high quality
known-face-images can be extracted from video file many times faster than using seperate
frame splitting/extraction/detection applications, or by methods that are CPU bound and only operate on individual images.

## Usage

```python
FG = FaceGrab('./images/nick-cage-reference')
if FG.has_references:
    FG.process('./movies/The Wicker Man.mp4', './extracted/nick-cage-wicker-man')
```

If run out of memory

1. Reduce the **batch_size**  - the whole thing will take longer
2. Decrease the process **scale**  e.g. 0.125 (1/8) - you may well get fewer face detections

If you are getting too many false positives 

1. Use a more varied, more representative, range of **reference**  images
2. Increase the **reference_jitter** so that each recognition is done using a higher number of resamples
3. Decrease the **tolerance** so that each recognition is stricter e.g. 0.4

If you are getting too few matches

1. Use a greater number/range of **reference** images (ideally ones that look like the person in the input)
2. Increase the **tolerance** so that each recognition is less strict e.g. 0.8
3. Decrease the **reference_jitter** so that each recognition is done fewer resamples (less accurate) 
4. Decrease the **skip_frames** amount so that more of the input is processed (this might result in very similar extracted images)
5. Increase the process **scale** e.g. 0.5 (1/2) - bearing in mind you may need to reduce the batch_size accordingly

### Memory  

Very roughly speaking batch_size * frame dimensions * process scale = VRAM needed


## Built using

- CUDA 8.0 - https://developer.nvidia.com/cuda-80-ga2-download-archive
- cudnn 6 for cuda 8 - https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170427/cudnn-8.0-windows10-x64-v6.0-zip (login required)
- dlib - https://github.com/davisking/dlib.git compiled with CUDA (and preferably AVX) see notes.
- Visual C++ 2015 Build Tools - http://landinghub.visualstudio.com/visual-cpp-build-tools

YMMV - pretty sure it would work just as well with CUDA 9 / cuDNN 7 / etc - but personally I could not get dlib to build with CUDA support against v9/9.1 :(
