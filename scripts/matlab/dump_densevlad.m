function dump_densevlad(out_dir)
%DUMP_DENSEVLAD Run the original DenseVLAD pipeline in MATLAB and save intermediates.
%  dump_densevlad() saves to ~/Library/Caches/densevlad/torii15/matlab_dump.
%  dump_densevlad(out_dir) saves to out_dir.

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

imfn = fullfile(root_dir, 'data', 'example_gsv', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg');
dictfn = fullfile(root_dir, 'data', 'dnscnt_RDSIFT_K128.mat');
pcafn = fullfile(root_dir, 'data', 'dnscnt_RDSIFT_K128_vlad_pcaproj.mat');

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

sizes = [4 6 8 10];
step = 2;
magnif = 6;
window_size = 1.5;
contrast_threshold = 0.005;
max_size = max(sizes);

for s = sizes
    off = floor(1.0 + 3.0 / 2.0 * (max_size - s));
    sigma = s / magnif;
    ims = vl_imsmooth(img_single, sigma);
    [f_s, d_s] = vl_dsift(ims, ...
        'Step', step, ...
        'Size', s, ...
        'Bounds', [off off +inf +inf], ...
        'Norm', ...
        'Fast', ...
        'WindowSize', window_size);
    [~, d_sf] = vl_dsift(ims, ...
        'Step', step, ...
        'Size', s, ...
        'Bounds', [off off +inf +inf], ...
        'Norm', ...
        'Fast', ...
        'WindowSize', window_size, ...
        'FloatDescriptors');
    contrast = f_s(3, :);
    d_s(:, contrast < contrast_threshold) = 0;
    d_sf(:, contrast < contrast_threshold) = 0;
    switch s
        case 4
            ims_4 = ims;
            f_4 = f_s;
            d_4 = d_s;
            d_4f = d_sf;
        case 6
            ims_6 = ims;
            f_6 = f_s;
            d_6 = d_s;
            d_6f = d_sf;
        case 8
            ims_8 = ims;
            f_8 = f_s;
            d_8 = d_s;
            d_8f = d_sf;
        case 10
            ims_10 = ims;
            f_10 = f_s;
            d_10 = d_s;
            d_10f = d_sf;
    end
end

[f, desc] = vl_phow(img_single);
desc_single = single(desc);
desc_rs = sqrt(bsxfun(@rdivide, desc_single, sum(abs(desc_single), 1) + 1e-12));

nn = vl_kdtreequery(kdtree, CX, desc_rs);
assigns = zeros(size(CX, 2), size(desc_rs, 2), 'single');
assigns(sub2ind(size(assigns), double(nn), 1:length(nn))) = 1;

vlad = vl_vlad(desc_rs, CX, assigns, 'NormalizeComponents');

load(pcafn, 'vlad_proj', 'vlad_lambda');
vladdim = 4096;
vlad_proj = single(vlad_proj(:, 1:vladdim)');
vlad_wht = diag(1 ./ sqrt(vlad_lambda(1:vladdim)));
v = vlad_wht * (vlad_proj * vlad);
v = single(v / norm(v));

out_path = fullfile(out_dir, 'densevlad_dump.mat');
save(out_path, ...
    'imfn', 'dictfn', 'pcafn', ...
    'img_gray', 'img_down', 'img_single', ...
    'sizes', 'step', 'magnif', 'window_size', 'contrast_threshold', ...
    'ims_4', 'ims_6', 'ims_8', 'ims_10', ...
    'f_4', 'f_6', 'f_8', 'f_10', ...
    'd_4', 'd_6', 'd_8', 'd_10', ...
    'd_4f', 'd_6f', 'd_8f', 'd_10f', ...
    'f', 'desc', 'desc_single', 'desc_rs', ...
    'nn', 'assigns', 'CX', 'vlad', 'v', ...
    '-v7.3');
fprintf(1, 'Saved: %s\n', out_path);
end
