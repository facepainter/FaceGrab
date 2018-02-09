Extract a known face from a video or image sequence.

This class uses a combination of a (precomputed) deep learning CNN model to batch detect faces
in video frames (in GPU with CUDA) then HoG to compare the detected faces with a reference set 
of "known" face encodings.

Using the GPU with CUDA in this way means batch processing face detection in up to 128 frames
at at time can be achieved (VRAM dependant). This combined with other speed/optimisation techniques
(such as downsampling, default frame skipping, etc) means that very high quality
known-face-images can be extracted from video file many hundreds of times faster than using seperate
frames spliting/extract/detect applications or methods that are CPU bound and only operate on individual images.

NB: requires dlib 19.9.99+ compiled with CUDA (and preferably AVX)
See: https://gist.github.com/facepainter/adfaabe25831a7c9300bafd1b886e1c8#file-dlib_avx_cuda-bat