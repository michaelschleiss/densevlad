24/7 place recognition by view synthesis
Akihiko Torii, Relja Arandjelović, Josef Sivic, Masatoshi Okutomi, Tomas Pajdla. 
CVPR 2015

Please send bug reports and suggestions to <torii@ctrl.titech.ac.jp>

-----------------------
GETTING STARTED
-----------------------
Open MATLAB (tested on R2015a on Mac-Yosemite & R2013b on Ubuntu-13.10-64bit) and run 
>> at_setup
>> test_densevlad % gives a Dense-SIFT VLAD vector
>> test_matching  % reproduces the results of feature matching shown in Figure 2 [Torii-CVPR-2015].
>> test_viewsynth % generates synthesized views using a Google streetview panorama and its depth-map.

-----------------------
CONTENT
-----------------------
This package contains MATLAB+MEX implementation of main functions from [Torii-CVPR-2015].

at_image2densevlad.m
	Describe Dense RootSIFT features and compute intra-normalized VLAD descriptor followed by a PCA compression to 4,096 dimensions.

at_gsv2viewsynth.m
	Generate synthesized views at user defined position and orientation using using a Google streetview panorama and its depth-map.

The code includes a pre-built 128D vocabulary for Dense-SIFT VLAD and a projection matrix for PCA compression. See [Torii-CVPR-2015] for the details of the experimental set-ups for each dataset.


-----------------------
CITATION
-----------------------
If you use this code, please cite:
 
@inproceedings{Torii-CVPR-2015,
author = {Torii, A. and Arandjelovi\'c, R. and Sivic, J. and Okutomi, M. and Pajdla, T.},
title = {24/7 place recognition by view synthesis},
booktitle = {CVPR},
year = {2015},
}


-----------------------
REQUIREMENTS
-----------------------
This package bundles two external packages. If you have a trouble for running this demo program, rebuild them.

(1) VLFeat 0.9.20
A. Vedaldi and B. Fulkerson

http://www.vlfeat.org/

(2) Library yael v438
Herve Jegou and Matthijs Douze. 

https://gforge.inria.fr/frs/download.php/31463/yael_v300.tar.gz

-----------------------
LICENSE
-----------------------
Copyright (c) 2015 Akihiko Torii

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
