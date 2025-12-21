function [oimg,olabel,lambda] = at_gsv2viewsynth(qR,qUTM,gsv,oimw,oimh,FoV)

R = qR*vl_rodr(gsv.heading/180*pi*[0; 1; 0]);
C = qUTM - gsv.utm(1:2,1);
T = -R*[C(2); 0; -C(1)];

pparam = gsv.param;
plabel = gsv.label;
img = gsv.img;

scale = size(img,2)/512;

%--- adjust the coordinates between new camera and GSV planes
CT = [...
  -1 0 0 0;
  0 0 1 0;
  0 -1 0 0;
  0 0 0 1];
pparam(2:5,:) = CT*pparam(2:5,:);

pN = pparam(2:4,2:end);
pD = pparam(5,2:end);

%--- convert image points to normalized unit vectors
K = [oimw/2*FoV 0 oimw/2; 0 oimw/2*FoV oimh/2; 0 0 1];
[gx, gy] = meshgrid(1:oimw,1:oimh);
gx = gx'; gy = gy';
X = [gx(:)'; gy(:)'; ones(1,oimw*oimh)];
Y = at_l2normalize_col(K\X);

psz1 = size(plabel,1);
psz2 = size(plabel,2);

if sum(abs(T)) > eps
  %--- compute the depth for all the planes
  lambda = bsxfun(@rdivide,pN'*(R'*T) - pD',pN'*(R'*Y));
  
  %--- retrieve the closest plane from the ones having a consistent label.
  x = -ones(1,oimw*oimh); y = x;
  
  d = at_synthreproj2(Y,lambda,R',T,plabel);
  
  d = [NaN(1,size(d,2)); d];
  [s,olabel] = min(d,[],1);
   
  inl = ~isnan(s);
  
  Z = bsxfun(@times,s(1,inl),Y(:,inl));
  Z = R'*bsxfun(@minus,Z,T);
  Z = at_l2normalize_col(Z);
  
  th = atan2(Z(1,:),Z(3,:));
  phi = asin(Z(2,:));
  [u,v] = at_sph2rct(th,phi,psz1,psz2);
  
  u = scale*u;
  v = scale*v;
  
  u(floor(u)==0) = 1;
  v(floor(v)==0) = 1;
  
  x(1,inl) = u;
  y(1,inl) = v;
  
else
  lambda = bsxfun(@rdivide,-pD',pN'*(R'*Y));
  
  Z = R'*Y;
  Z = at_l2normalize_col(Z);
  
  th = atan2(Z(1,:),Z(3,:));
  phi = asin(Z(2,:));
  [u,v] = at_sph2rct(th,phi,psz1,psz2);
  x = scale*u;
  y = scale*v;
  
  x(floor(x)==0) = 1;
  y(floor(y)==0) = 1;
  
  uu = round(u);
  uu(uu==0) = 1;
  uu(uu>512) = 512;
  
  vv = round(v);
  vv(vv==0) = 1;
  vv(vv>256) = 256;
    
  sidx = sub2ind(size(plabel),uu,vv);
  olabel = plabel(sidx)+1;
end

%--- image synthesis with bilinear interpolation
bilinear = 1;
pts = []; pts.x = reshape(x,oimw,oimh)'; pts.y = reshape(y,oimw,oimh)';
oimg = mh_iminterp(img,pts,bilinear);
