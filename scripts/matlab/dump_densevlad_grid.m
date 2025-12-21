function dump_densevlad_grid(out_dir)
%DUMP_DENSEVLAD_GRID Dump masked DenseVLAD intermediates for example_grid.
%  dump_densevlad_grid() saves to ~/Library/Caches/densevlad/torii15/matlab_dump.
%  dump_densevlad_grid(out_dir) saves to out_dir.

if nargin < 1 || isempty(out_dir)
    out_dir = fullfile(getenv('HOME'), 'Library', 'Caches', 'densevlad', 'torii15', 'matlab_dump');
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

cache_dir = fullfile(getenv('HOME'), 'Library', 'Caches', 'densevlad', 'torii15');
zip_path = fullfile(cache_dir, '247code.zip');
root_dir = fullfile(cache_dir, '247code');
url = 'http://www.ok.ctrl.titech.ac.jp/~torii/project/247/download/247code.zip';

if ~exist(zip_path, 'file')
    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
    fprintf(1, 'Downloading %s to %s\n', url, zip_path);
    websave(zip_path, url);
end

vl_setup_path = fullfile(root_dir, 'thirdparty', 'vlfeat-0.9.20', 'toolbox', 'vl_setup.m');
if ~exist(vl_setup_path, 'file')
    fprintf(1, 'Extracting %s to %s\n', zip_path, cache_dir);
    unzip(zip_path, cache_dir);
end

addpath(root_dir);
if exist(vl_setup_path, 'file')
    run(vl_setup_path);
else
    error('VLFeat setup not found: %s', vl_setup_path);
end
addpath(fullfile(root_dir, 'code'));

imfn = fullfile(root_dir, 'data', 'example_grid', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg');
dictfn = fullfile(root_dir, 'data', 'dnscnt_RDSIFT_K128.mat');
labelfn = fullfile(root_dir, 'data', 'example_grid', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_000.label.mat');
planefn = fullfile(root_dir, 'data', 'example_grid', 'planes.txt');

load(dictfn, 'CX');
CX = bsxfun(@rdivide, CX, sqrt(sum((CX.^2), 1)));
kdtree = vl_kdtreebuild(CX);

img = imread(imfn);
if size(img, 3) > 1
    img_gray = rgb2gray(img);
else
    img_gray = img;
end
img_down = vl_imdown(img_gray);
img_single = single(img_down);
if isinteger(img_down)
    img_single = img_single ./ single(intmax(class(img_down)));
end

[f, desc] = vl_phow(img_single);
desc = relja_rootsift(single(desc));

% mask areas damaged by synthesis (same as at_grid_maskfeatures_dense)
r = load(labelfn);
label = r.label;

fp = fopen(planefn, 'r');
fgetl(fp);
A = fscanf(fp, '%f %f %f %f %f\n');
planes = reshape(A, 5, []);
fclose(fp);

np = planes(2:4, :);
nz = [0 0 -1]';
ang = acos(nz' * np);
pmask = find(ang < 20 / 180 * pi);
for ff = pmask
    label(label == ff) = -1;
end

plabel = reshape(label, size(img_down, 2), size(img_down, 1))';
bmsk = plabel < 2;

scl = max(f(4, :)) * 2;
[xg, yg] = meshgrid(-scl:scl, -scl:scl);
disk = (xg.^2 + yg.^2) <= scl^2;
ebmsk = conv2(double(bmsk), double(disk), 'same') > 0;

x = round(f(1, :));
y = round(f(2, :));
idx = sub2ind(size(ebmsk), y, x);
msk = ebmsk(idx) > 0;

f_masked = f(:, ~msk);
desc_masked = desc(:, ~msk);

nn = vl_kdtreequery(kdtree, CX, desc_masked);
assigns = zeros(size(CX, 2), size(desc_masked, 2), 'single');
assigns(sub2ind(size(assigns), double(nn), 1:length(nn))) = 1;
vlad = vl_vlad(desc_masked, CX, assigns, 'NormalizeComponents');

out_path = fullfile(out_dir, 'densevlad_grid_dump.mat');
save(out_path, ...
    'imfn', 'dictfn', 'labelfn', 'planefn', ...
    'img_gray', 'img_down', 'img_single', ...
    'f', 'desc', 'f_masked', 'desc_masked', 'msk', ...
    'CX', 'vlad', ...
    '-v7.3');
fprintf(1, 'Saved: %s\n', out_path);
end
