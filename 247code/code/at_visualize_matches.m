function at_visualize_matches(imfn,f,matches,inliers)

figure; 
set(gcf, 'Position', [4 24 360 240]);
imshow(rgb2gray(imread(imfn)),'initialmagnification',20); 
colormap('gray'); hold on;
plot(f(1,matches(1,:)),f(2,matches(1,:)),'r.');
plot(f(1,inliers(1,:)),f(2,inliers(1,:)),'g.');