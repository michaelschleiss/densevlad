function [matches, inliers, f1, f2, Iratio] = at_SIFT_matching(imfn1,imfn2,sifttype,labelfn2)

%--- feature detection & description
fprintf(1,'  Detection and description ...');

img1 = imread(imfn1);
if (size(img1,3)>1), img1 = rgb2gray(img1); end;   
switch sifttype 
  case 'DSIFT'
    [f1, desc1] = vl_phow(im2single(vl_imdown(img1)));
    f1([1; 2; 4],:) = 2*f1([1; 2; 4],:); % hack for visualization
  case 'SIFT'
    [f1, desc1] = vl_covdet(im2single(img1),'Method','DoG','descriptor','SIFT',...
      'EstimateAffineShape',false, 'EstimateOrientation', false, 'DoubleImage',false);
end
desc1 = relja_rootsift(single(desc1));    


img2 = imread(imfn2);
if (size(img2,3)>1), img2 = rgb2gray(img2); end;   
if strcmp(sifttype,'DSIFT')
  [f2, desc2] = vl_phow(im2single(vl_imdown(img2)));
  f2([1; 2; 4],:) = 2*f2([1; 2; 4],:); % hack for visualization
else
  [f2, desc2] = vl_covdet(im2single(img2),'Method','DoG','descriptor','SIFT',...
    'EstimateAffineShape',false, 'EstimateOrientation', false, 'DoubleImage',false);
end
desc2 = relja_rootsift(single(desc2)); 


if nargin > 3
  [f2, desc2] = at_grid_maskfeatures_dense(f2,desc2,size(img2),labelfn2);
end
desc2 = single(desc2);

%--- simple NN matching
fprintf(1,'  Done.\n  NN (may take a few minites) ...');
[matches,idx12,idx21] = at_NN_match(f1,desc1,f2,desc2,sifttype);

%--- ransacing 
fprintf(1,'  Done.\n  Ransacing (may take a few minites)... ');
inliers = cell(1,5);
origindex = 1:size(matches,2);
mt = matches;
ii = 0;
while ii<5
  ii=ii+1;
  [H, inls] = at_ransacH4(f1(1:2,mt(1,:)), f2(1:2,mt(2,:)), 10000, 10, 0, .99);
  if length(inls) > 15
    inliers{ii} = origindex(1,inls);
    mt(:,inls) = [];
    origindex(:,inls) = [];
  else
    break
  end;
end
tmp = [inliers{:}];
inliers = matches(1:2,tmp);
fprintf(1,'  Done.\n  Total: matches %d, inliers %d\n',size(matches,2),size(inliers,2));
Iratio = size(inliers,2) / size(matches,2);
matches(:,tmp) = [];


