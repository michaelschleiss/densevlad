%--- A script for reproducing figure 2 in our CVPR15 paper.

clear all; close all;

pth = './data/matches/';

imfn1 = [pth 'query.jpg'];
imfn2 = [pth 'gsv.jpg'];
imfn3 = [pth 'synth.jpg'];

labelfn2 = [pth 'synth.label.mat'];

vskip = 1000;

display('Matching SIFT between query and street-view ...')
[matches, inliers, f1, f2, Iratio] = at_SIFT_matching(imfn1,imfn2,'SIFT');
%--- visualization
if size(matches,2) > vskip, matches = matches(:,randperm(size(matches,2),vskip)); end
at_visualize_matches(imfn1,f1,matches(1,:),inliers(1,:));
set(gcf, 'Position', [ 50 300 240 160]); set(gcf,'Name','Fig2(b) Sparse SIFT','NumberTitle','off')
at_visualize_matches(imfn2,f2,matches(2,:),inliers(2,:));
set(gcf, 'Position', [300 300 240 160]); set(gcf,'Name',sprintf('Inlier ratio=%.2f',Iratio),'NumberTitle','off')
fprintf(1,'Done.\n\n'); 


display('Matching SIFT between query and synthesized view...')
[matches, inliers, f1, f2, Iratio] = at_SIFT_matching(imfn1,imfn3,'SIFT',labelfn2);
%--- visualization
if size(matches,2) > vskip, matches = matches(:,randperm(size(matches,2),vskip)); end
at_visualize_matches(imfn1,f1,matches(1,:),inliers(1,:));
set(gcf, 'Position', [550 300 240 160]); set(gcf,'Name','Fig2(e) Sparse SIFT','NumberTitle','off')
at_visualize_matches(imfn3,f2,matches(2,:),inliers(2,:));
set(gcf, 'Position', [800 300 240 160]); set(gcf,'Name',sprintf('Inlier ratio=%.2f',Iratio),'NumberTitle','off')
fprintf(1,'Done.\n\n'); 


display('Matching Dense SIFT between query and street-view ...')
[matches, inliers, f1, f2, Iratio] = at_SIFT_matching(imfn1,imfn2,'DSIFT');
%--- visualization
if size(matches,2) > vskip, matches = matches(:,randperm(size(matches,2),vskip)); end
at_visualize_matches(imfn1,f1,matches(1,:),inliers(1,:));
set(gcf, 'Position', [ 50 80 240 160]); set(gcf,'Name','Fig2(c) Dense SIFT','NumberTitle','off')
at_visualize_matches(imfn2,f2,matches(2,:),inliers(2,:));
set(gcf, 'Position', [300 80 240 160]); set(gcf,'Name',sprintf('Inlier ratio=%.2f',Iratio),'NumberTitle','off')
fprintf(1,'Done.\n\n'); 


display('Matching Dense SIFT between query and synthesized view...')
[matches, inliers, f1, f2, Iratio] = at_SIFT_matching(imfn1,imfn3,'DSIFT',labelfn2);
%--- visualization
if size(matches,2) > vskip, matches = matches(:,randperm(size(matches,2),vskip)); end
at_visualize_matches(imfn1,f1,matches(1,:),inliers(1,:));
set(gcf, 'Position', [550 80 240 160]); set(gcf,'Name','Fig2(f) Dense SIFT','NumberTitle','off')
at_visualize_matches(imfn3,f2,matches(2,:),inliers(2,:));
set(gcf, 'Position', [800 80 240 160]); set(gcf,'Name',sprintf('Inlier ratio=%.2f',Iratio),'NumberTitle','off')
fprintf(1,'Done.\n\n'); 


