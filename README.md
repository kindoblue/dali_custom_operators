## Custom operators for DALI
Here you will find two custom operators for DALI:
 * a modified file reader that supports segmentation scenario
 * a modified TIFF reader that supports compressed minisblack's images
 
For more information, there is this blog post:
https://www.kindoblue.nl/custom-dali-operators/  

### Installation

###### Prerequisites
After having installed CUDA you should create a virtualenv and install the dependencies. 
Like this:
 * `mkvirtualenv dali -ppython3`
 * `workon dali`
 * `export CUDA_HOME=/usr/local/cuda-11.0`  ⟵ here your CUDA dir
 * `export CFLAGS="-I$CUDA_HOME/include`
 * `pip install  -r requirements.txt`

###### Build
 * `workon dali` 
 * `mkdir cmake-build-debug && cmake-build-debug`
 * `cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME ..`
 * `make -j4`
 
### Other 

To debug the plugin with gdb

```
gdb python
break main
run test.py
## breakpoint 1 is hit
set stop-on-solib 1
cont
## stops when any new shared library is loaded
info shared
```