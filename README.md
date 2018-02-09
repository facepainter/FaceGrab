Extract a known face from a video or image sequence.

This class uses a combination of a deep learning CNN model to batch detect faces
in video frames (in GPU with CUDA) then HoG to compare faces with a pre existing
reference set of face encodings.

Using the GPU with CUDA in this way means batch processing face detection in up to 128 frames
at at time can be achieved (VRAM dependant). This combined with other speed/optimisation techniques
(such as downsampling, default frame skipping, etc) means that very high quality
known-face-images can be extracted from video file many hundreds of times faster than using seperate
frames spliting/extract/detect applications or methods that are CPU bound and only operate on individual images.

NB: requires dlib 19.9.99+ compiled with CUDA (and preferably AVX)
see  