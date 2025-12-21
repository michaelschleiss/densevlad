clear all; close all;

pth = './data/';
gpid = 'L-NLvGeZ6JHX6JO8Xnf_BA'; % Google Street-View (GSV) panorama ID

outpth = './data/synth/'; % results will be saved here.
at_dirgen(outpth);

%--- output image size (larger slower)
oimw = 640; % 1280;
oimh = 480; % 960;

FoV = 1/tan(28.4157/180*pi); 
pitch = 12; % degrees 
yaw = (0:11)/12*360;  % interval for yaw

%--- generate synth views
fprintf(1,'Loading a Google street-view panorama and its depth-map ... \n');

gsv = at_readgsvfiles(sprintf('%s%s/',pth,gpid));
load(sprintf('%s%s/utm.mat',pth,gpid),'utm');
gsv.utm = utm;

qutm = [utm(1) + 2; utm(2)]; % position of synthesized view in UTM, e.g. 
                             % 5 meters in x-coords from GSV position.
                             % try qutm=utm to generate cutouts with no
                             % synth.

figure; imagesc(gsv.img); axis image; axis off;
set(gcf,'Name','Street-view panorama','NumberTitle','off');
set(gcf, 'Position', [50 600 360 180]);

D = uint8(255*(gsv.depth'./max(gsv.depth(:))));
figure; image(D); axis image; axis off; colormap('gray');
set(gcf,'Name','Associated depth-map','NumberTitle','off');
set(gcf, 'Position', [420 600 360 180]);

figure; image(gsv.label'); axis image; axis off; colormap('Lines');
set(gcf,'Name','Individual scene planes','NumberTitle','off');
set(gcf, 'Position', [790 600 360 180]);

fprintf(1,'Done.\nGenerating synthesized views at the position of \n(%e, %e) \n',qutm(1),qutm(2));
fprintf(1,'using street-view panorama at the position of \n(%e, %e) in UTM. \n\n',utm(1),utm(2));  

hv = figure; set(gcf, 'Position', [50 80 360 240]); 
      
flt = ones(9); % fill small holes. 
for ii=1:length(pitch)
  for jj=1:length(yaw)
    imfn    = [outpth sprintf('%s_%03d_%03d.jpg',gpid,pitch(ii),yaw(jj))];
    labelfn = [outpth sprintf('%s_%03d_%03d.label.mat',gpid,pitch(ii),yaw(jj))];
    depthfn = [outpth sprintf('%s_%03d_%03d.depth.mat',gpid,pitch(ii),yaw(jj))];
    if ~exist(imfn,'file') 
      fprintf(1,'  for pitch %03d deg, yaw=%03d deg\n',pitch(ii),yaw(jj));
      
      qR = vl_rodr(pitch(ii)/180*pi*[-1;0;0]) * vl_rodr(yaw(jj)/180*pi*[0;-1;0]);
            
      [oimg,label,lambda] = at_gsv2viewsynth(qR,qutm,gsv,oimw,oimh,FoV);
      at_dirgen(imfn);
      img = rgb2gray(oimg)';
        
      idx = find(label==1);
      A = sparse(idx,ones(1,length(idx)),1,oimw*oimh,1);
      B = 1-full(reshape(A,oimw,oimh));
      C = imfilter(B,flt);      
      img2 = imfilter(double(img),flt);
      idx2 = idx(C(idx) > 0);        
      D = round(img2(idx2)./C(idx2));
      img(idx2) = D;
      imwrite(img',imfn,'Quality',90);
      clear A B C D
      
      lambda2 = [100*ones(1,size(lambda,2)); lambda];
      A = sparse(label,1:length(label),1,size(lambda2,1),length(label));
      D = A.*lambda2;
      [ix,iy,iz] = find(D);
      D = single(reshape(iz,[oimw oimh]));
      save('-v6',depthfn,'D');
      clear A D
      
      L = reshape(label,oimw,oimh);
      L3 = ordfilt2(L,81,flt);      
      label(idx2) = round(L3(idx2));      
      label = uint16(label);
      save('-v7.3',labelfn,'label');      
            
      figure(hv); clf; 
      set(gcf, 'Position', [50 80 360 240]); 
      imagesc(img'); axis image; axis off; colormap('gray'); drawnow;
      set(gcf,'Name',sprintf('pitch=%2.2f, yaw=%2.2f',pitch(ii),yaw(jj)),'NumberTitle','off')
    end
  end
end

fprintf('The synthesized views are saved in %s \n',outpth);
fprintf('Associated depth-map and plane labels are also saved there.');
