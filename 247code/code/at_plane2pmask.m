function [pmask,Nplanes] = at_plane2pmask(plane_fn)


%--- read planes
% plane_fn = [opt.pth sprintf('gsv/%s/planes.txt',gpid)];
fp = fopen(plane_fn,'r');
tline = fgetl(fp);
Nplanes = sscanf(tline,'%d');
A = fscanf(fp,'%f %f %f %f %f\n');
planes = reshape(A,5,[]);
fclose(fp);

np = planes(2:4,:);
nz = [0 0 -1]';
ang = acos(nz'*np);
pmask = find(ang < 20/180*pi);
