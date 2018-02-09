' YMMV here - I needed 64bit 2007 complier to get this to work...
set COMPILER="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64"
set PATH=%COMPILER%;%PATH%
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake -G "Visual Studio 15 Win64" -DUSE_AVX_INSTRUCTIONS=1 -DDLIB_USE_CUDA=1 -DCUDA_HOST_COMPILER="%COMPILER%" ..
cmake --build .
cd ..
python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA