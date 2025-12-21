function vlad = at_image2densevlad(imfn,dictfn,planefn,labelfn)

%--- load visual vocabulary (centroids)
load(dictfn,'CX');
CX = at_l2normalize_col(CX);
kdtree= vl_kdtreebuild(CX);

%--- read image, convert gray, downsize to vga
img = imread(imfn);
if (size(img,3)>1)
  img=rgb2gray(img);
end;   
img = vl_imdown(img);

[f, desc] = vl_phow(im2single(img));
desc = relja_rootsift(single(desc));    

%--- mask areas damaged by synthesis
if nargin > 3
  [f, desc] = at_grid_maskfeatures_dense(f,desc,size(img),labelfn,planefn);
end

vlad = relja_computeVLAD(desc, CX, kdtree);

% figure; 
% imshow(img); hold on;
% plot(f(1,:),f(2,:),'b.');
