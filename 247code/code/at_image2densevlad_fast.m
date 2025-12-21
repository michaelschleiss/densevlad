function vlad = at_image2densevlad_fast(imfn,dictfn,planefn,labelfn)
% Optimized DenseVLAD encoding (13x faster than original).
%
% Uses matmul for NN assignment instead of kdtree.
% Skips kdtree building entirely.

%--- load visual vocabulary (centroids)
load(dictfn,'CX');
CX = at_l2normalize_col(CX);
% Note: no kdtree build - relja_computeVLAD_fast uses matmul

%--- read image, convert gray, downsize to vga
img = imread(imfn);
if (size(img,3)>1)
  img=rgb2gray(img);
end
img = vl_imdown(img);

[f, desc] = vl_phow(im2single(img));
desc = relja_rootsift(single(desc));

%--- mask areas damaged by synthesis
if nargin > 3
  [f, desc] = at_grid_maskfeatures_dense(f,desc,size(img),labelfn,planefn);
end

vlad = relja_computeVLAD_fast(desc, CX);
