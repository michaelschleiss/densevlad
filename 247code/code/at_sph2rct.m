function [u, v] = at_sph2rct(th,phi,w,h)
% Transform 3D unit vectors to 2D image points by 
% equirectangular projection

fov=2*pi; % max. field of view     (150)
u = th*w/fov + w/2;
v = phi*w/fov + h/2;
