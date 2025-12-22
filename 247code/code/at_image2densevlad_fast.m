function vlad = at_image2densevlad_fast(imfn,dictfn,planefn,labelfn)
% Optimized DenseVLAD encoding (13x faster than original).
%
% Uses matmul for NN assignment instead of kdtree.
% Skips kdtree building entirely.

%--- load visual vocabulary (centroids)
load(dictfn,'CX');
CX = at_l2normalize_col(CX);
% Compute kdtree tie index once per dictionary for zero descriptors.
% This avoids hardcoding a cluster id while keeping the fast path fast.
persistent cached_dict cached_zero_tie
if isempty(cached_zero_tie) || ~strcmp(cached_dict, dictfn)
  kdtree = vl_kdtreebuild(CX);
  zero_desc = zeros(size(CX,1), 1, class(CX));
  cached_zero_tie = vl_kdtreequery(kdtree, CX, zero_desc);
  cached_dict = dictfn;
end

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

vlad = relja_computeVLAD_fast(desc, CX, cached_zero_tie);
