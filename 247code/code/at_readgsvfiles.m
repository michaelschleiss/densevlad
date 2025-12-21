function gsv = at_readgsvfiles(gpth,loadimg)
if nargin<2, loadimg=true; end

gsv = [];

label_fn = [gpth 'labels.txt'];
plane_fn = [gpth 'planes.txt'];
image_fn = [gpth 'panorama.jpg'];
gps_fn   = [gpth 'gps.txt'];

%--- read image
if loadimg
  gsv.img = imread(image_fn);
else
  gsv.img = [];
end

%--- read depth map labels
fp = fopen(label_fn,'r');
tline = fgetl(fp);
A = sscanf(tline,'%d,%d');
tline = fgetl(fp);
fclose(fp);
% B = str2num(tline);
B = sscanf(tline,'%d');
gsv.label = reshape(B,A(1),A(2));
clear A B

%--- read depth map
fp = fopen(label_fn,'r');
tline = fgetl(fp);
A = sscanf(tline,'%d,%d');
tline = fgetl(fp);
fclose(fp);
% B = str2num(tline);
B = sscanf(tline,'%f');
gsv.depth = reshape(B,A(1),A(2));
clear A B

%--- read planes
fp = fopen(plane_fn,'r');
tline = fgetl(fp);
Nplanes = sscanf(tline,'%d');
A = fscanf(fp,'%f %f %f %f %f\n');
gsv.param = reshape(A,5,[]);
fclose(fp);
clear A B

%--- read GPS position and heading angle of gsv panorama
gps = zeros(2,1); 
heading = zeros(1,1);
fp = fopen(gps_fn,'r');
tline = fgetl(fp);
tmp = strfind(tline,' ');
gps(1,1) = str2double(tline(1:tmp(1)-1));
gps(2,1) = str2double(tline(tmp(1)+1:tmp(2)-1));
heading(1) = str2double(tline(tmp(2)+1:tmp(3)-1));
fclose(fp);
gsv.gps = gps;
gsv.heading = heading;
