function [f, desc] = at_grid_maskfeatures_dense(f,desc,sz,labelfn,planefn)
% create mask for not describing areas on holes.

if ~isempty(f)
  %--- load plane label per pixel
  r = load(labelfn);
  label = r.label;
  
  %--- read planes
  if nargin > 4
    fp = fopen(planefn,'r');
    tline = fgetl(fp);
    nplanes = sscanf(tline,'%d');
    A = fscanf(fp,'%f %f %f %f %f\n');
    planes = reshape(A,5,[]);
    fclose(fp);
    
    np = planes(2:4,:);
    nz = [0 0 -1]';
    ang = acos(nz'*np);
    pmask = find(ang < 20/180*pi);
    
    for ff=pmask
      label(label==ff) = -1;
    end
  end
  
  plabel = reshape(label,sz(2),sz(1))';
  bmsk = plabel < 2;
  
  scl = max(f(4,:))*2;
  
  se = strel('disk',scl);
  ebmsk = imdilate(bmsk,se);
  
  msk = zeros(1,size(f,2));
  x = round(f(1,:));
  y = round(f(2,:));
  idx = sub2ind(size(ebmsk),y,x);
  msk(ebmsk(idx) > 0) = 1;
  
  f = f(:,~msk);
  desc = desc(:,~msk);
  
  
  %   for ff=pmask
  %     label(label==ff) = -1;
  %   end
  %   plabel = reshape(label,sz(2),sz(1))';
  %   x = round(f(1,:));
  %   y = round(f(2,:));
  %   idx = sub2ind(size(plabel),y,x);
  %   msk = plabel(idx) < 0;
  %   f = f(:,~msk);
  %   desc = desc(:,~msk);
  %
  %   msk = zeros(1,size(f,2));
  %
  %   x = round(f(1,:));
  %   y = round(f(2,:));
  %   x = max(1,x); x = min(x,sz(2)); y = max(1,y); y = min(y,sz(1));
  %   idx = sub2ind(size(plabel),y,x);
  %   msk(plabel(idx) < 2) = 1;
  %
  %
  %   x = round(f(1,:) + 2*f(4,:));
  %   y = round(f(2,:) + 2*f(4,:));
  %   x = max(1,x); x = min(x,sz(2)); y = max(1,y); y = min(y,sz(1));
  %   idx = sub2ind(size(plabel),y,x);
  %   msk(plabel(idx) < 2) = 1;
  %
  %   x = round(f(1,:) + 2*f(4,:));
  %   y = round(f(2,:) - 2*f(4,:));
  %   x = max(1,x); x = min(x,sz(2)); y = max(1,y); y = min(y,sz(1));
  %   idx = sub2ind(size(plabel),y,x);
  %   msk(plabel(idx) < 2) = 1;
  %
  %   x = round(f(1,:) - 2*f(4,:));
  %   y = round(f(2,:) + 2*f(4,:));
  %   x = max(1,x); x = min(x,sz(2)); y = max(1,y); y = min(y,sz(1));
  %   idx = sub2ind(size(plabel),y,x);
  %   msk(plabel(idx) < 2) = 1;
  %
  %   x = round(f(1,:) - 2*f(4,:));
  %   y = round(f(2,:) - 2*f(4,:));
  %   x = max(1,x); x = min(x,sz(2)); y = max(1,y); y = min(y,sz(1));
  %   idx = sub2ind(size(plabel),y,x);
  %   msk(plabel(idx) < 2) = 1;
  %
  %   f = f(:,~msk);
  %   desc = desc(:,~msk);
  
end
