**FaceGrab**

Extract a known face from a video or image sequence.

Uses a combination of a (precomputed) deep learning CNN model to quickly batch detect faces
in video frames (upto 128 at a time in GPU with CUDA) then HoG face recognition with a computed
reference face encoding or set of encodings.

Using the GPU with CUDA in this way means batch processing face detection in up to 128 frames
at at time can be achieved (VRAM dependant). This combined with other speed/optimisation techniques
(such as downsampling, default frame skipping, etc) means that very high quality
known-face-images can be extracted from video file many hundreds of times faster than using seperate
frames spliting/extract/detect applications or methods that are CPU bound and only operate on individual images.

**Requires:**

- CUDA 8.0 - https://developer.nvidia.com/cuda-80-ga2-download-archive
- cudnn 6 for cuda 8 - https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170427/cudnn-8.0-windows10-x64-v6.0-zip (login required)
- dlib - https://github.com/davisking/dlib.git compiled with CUDA (and preferably AVX) see notes.
- Visual C++ 2015 Build Tools - http://landinghub.visualstudio.com/visual-cpp-build-tools