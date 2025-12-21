function dump_densevlad_all_intermediate(mode, varargin)
%DUMP_DENSEVLAD_ALL_INTERMEDIATE Run DenseVLAD MATLAB dumps (with intermediates).
%  dump_densevlad_all_intermediate('densevlad') writes densevlad_dump_intermediate.mat
%  dump_densevlad_all_intermediate('tokyo247', ...) writes Tokyo247 golden references
%  dump_densevlad_all_intermediate('all') runs all of the above.

if nargin < 1 || isempty(mode)
    mode = 'all';
end
mode = lower(mode);

switch mode
    case 'densevlad'
        run_densevlad(varargin{:});
    case 'tokyo247'
        run_tokyo247_golden(varargin{:});
    case 'all'
        run_densevlad(varargin{:});
        run_tokyo247_golden(varargin{:});
    otherwise
        error('Unknown mode: %s', mode);
end
end

function run_densevlad(out_dir)
%  run_densevlad() saves to ./assets/torii15/matlab_dump.
%  run_densevlad(out_dir) saves to out_dir.

if nargin < 1 || isempty(out_dir)
    out_dir = default_out_dir();
end
ensure_dir(out_dir);

root_dir = ensure_root_dir();
orig_dir = pwd;
cd(root_dir);
run('at_setup');
cd(orig_dir);
try
    vl_setnumthreads(1);
catch
end

imfn = fullfile(root_dir, 'data', 'example_gsv', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg');
imfn_030 = fullfile(root_dir, 'data', 'example_gsv', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_030.jpg');
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

vlad = at_image2densevlad(imfn, dictfn);
vlad_030 = at_image2densevlad(imfn_030, dictfn);

load(pcafn, 'vlad_proj', 'vlad_lambda');
vladdim = 4096;
vlad_proj = single(vlad_proj(:, 1:vladdim)');
vlad_wht = diag(1 ./ sqrt(vlad_lambda(1:vladdim)));
v = single(yael_vecs_normalize(vlad_wht * (vlad_proj * vlad)));

out_path = fullfile(out_dir, 'densevlad_dump_intermediate.mat');
save(out_path, ...
    'imfn', 'imfn_030', 'dictfn', 'pcafn', ...
    'img_gray', 'img_down', 'img_single', ...
    'sizes', 'step', 'magnif', 'window_size', 'contrast_threshold', ...
    'ims_4', 'ims_6', 'ims_8', 'ims_10', ...
    'f_4', 'f_6', 'f_8', 'f_10', ...
    'd_4', 'd_6', 'd_8', 'd_10', ...
    'd_4f', 'd_6f', 'd_8f', 'd_10f', ...
    'f', 'desc', 'desc_single', 'desc_rs', ...
    'nn', 'assigns', 'CX', 'vlad', 'v', 'vlad_030', ...
    '-v7.3');
fprintf(1, 'Saved: %s\n', out_path);
end

function run_tokyo247_golden(varargin)
%  run_tokyo247_golden() writes Tokyo247 golden refs to matlab_dump cache.

opts = struct();
opts.seed = 1337;
opts.num_db = 5;
opts.num_q = 5;
opts.max_dim = 640;
opts.use_imdown = false;
opts.out_dir = '';
opts = parse_opts(opts, varargin{:});

if isempty(opts.out_dir)
    opts.out_dir = default_out_dir();
end
ensure_dir(opts.out_dir);

root_dir = ensure_root_dir();
orig_dir = pwd;
cd(root_dir);
run('at_setup');
cd(orig_dir);
cache_dir = default_cache_dir();
tokyo_root = fullfile(cache_dir, 'tokyo247');
db_dir = fullfile(tokyo_root, 'database_gsv_vga');
query_dir = fullfile(cache_dir, '247query_subset_v2');
dbstruct_path = fullfile(tokyo_root, 'tokyo247.mat');

dictfn = fullfile(root_dir, 'data', 'dnscnt_RDSIFT_K128.mat');
pcafn = fullfile(root_dir, 'data', 'dnscnt_RDSIFT_K128_vlad_pcaproj.mat');

fprintf(1, 'Loading vocabulary...\n');
load(dictfn, 'CX');
CX = bsxfun(@rdivide, CX, sqrt(sum(CX.^2, 1)));
kdtree = vl_kdtreebuild(CX);

fprintf(1, 'Loading PCA...\n');
load(pcafn, 'vlad_proj', 'vlad_lambda');
vladdim = 4096;
vlad_proj = single(vlad_proj(:, 1:vladdim)');
vlad_wht = single(diag(1 ./ sqrt(vlad_lambda(1:vladdim))));

golden_list = fullfile(opts.out_dir, 'tokyo247_golden_list.txt');
if exist(golden_list, 'file') == 2
    fprintf(1, 'Loading golden list...\n');
    [sel_db_paths, sel_db_rel, sel_q_paths, sel_q_rel] = read_golden_list(golden_list, db_dir, query_dir);
    opts.num_db = numel(sel_db_paths);
    opts.num_q = numel(sel_q_paths);
    paths = [sel_db_paths; sel_q_paths];
    rel_paths = [sel_db_rel; sel_q_rel];
    is_db = [true(opts.num_db, 1); false(opts.num_q, 1)];
else
    fprintf(1, 'Loading dbStruct...\n');
    dbs = load(dbstruct_path, 'dbStruct');
    dbStruct = dbs.dbStruct;

    if isfield(dbStruct, 'dbImage')
        db_images = dbStruct.dbImage(:);
    elseif isfield(dbStruct, 'dbImageFns')
        db_images = dbStruct.dbImageFns(:);
    else
        error('dbStruct missing dbImage/dbImageFns');
    end

    if isfield(dbStruct, 'qImage')
        q_images = dbStruct.qImage(:);
    elseif isfield(dbStruct, 'qImageFns')
        q_images = dbStruct.qImageFns(:);
    else
        error('dbStruct missing qImage/qImageFns');
    end

    db_paths = cellfun(@(s) fullfile(db_dir, strrep(s, '.jpg', '.png')), db_images, 'UniformOutput', false);
    db_exists = cellfun(@(p) exist(p, 'file') == 2, db_paths);
    db_images = db_images(db_exists);
    db_paths = db_paths(db_exists);

    q_paths = cellfun(@(s) fullfile(query_dir, s), q_images, 'UniformOutput', false);
    q_exists = cellfun(@(p) exist(p, 'file') == 2, q_paths);
    q_images = q_images(q_exists);
    q_paths = q_paths(q_exists);

    if numel(db_paths) < opts.num_db
        error('Not enough DB images found (have %d, need %d).', numel(db_paths), opts.num_db);
    end
    if numel(q_paths) < opts.num_q
        error('Not enough query images found (have %d, need %d).', numel(q_paths), opts.num_q);
    end

    rng(opts.seed, 'twister');
    db_idx = randperm(numel(db_paths), opts.num_db);
    q_idx = randperm(numel(q_paths), opts.num_q);

    sel_db_paths = db_paths(db_idx);
    sel_db_rel = db_images(db_idx);
    sel_q_paths = q_paths(q_idx);
    sel_q_rel = q_images(q_idx);

    paths = [sel_db_paths; sel_q_paths];
    rel_paths = [sel_db_rel; sel_q_rel];
    is_db = [true(opts.num_db, 1); false(opts.num_q, 1)];
end
total = numel(paths);

vlad_pre = zeros(16384, total, 'single');
vlad_4096 = zeros(4096, total, 'single');

fprintf(1, 'Computing DenseVLAD for %d images...\n', total);
for i = 1:total
    imfn = paths{i};
    img = imread(imfn);
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    img = resize_max_dim(img, opts.max_dim);
    if opts.use_imdown
        img = vl_imdown(img);
    end
    img_single = single(img);
    if isinteger(img)
        img_single = img_single ./ single(intmax(class(img)));
    end
    [~, desc] = vl_phow(img_single);
    desc = relja_rootsift(single(desc));
    vlad = relja_computeVLAD(desc, CX, kdtree);
    vlad_pre(:, i) = single(vlad);
    v = yael_vecs_normalize(vlad_wht * (vlad_proj * vlad));
    vlad_4096(:, i) = single(v);
    fprintf(1, '  %d/%d\n', i, total);
end

max_dim = opts.max_dim;
use_imdown = opts.use_imdown;
out_mat = fullfile(opts.out_dir, 'tokyo247_golden.mat');
out_list = fullfile(opts.out_dir, 'tokyo247_golden_list.txt');
save(out_mat, 'vlad_pre', 'vlad_4096', 'is_db', 'rel_paths', ...
    'opts', 'max_dim', 'use_imdown', '-v7.3');

fid = fopen(out_list, 'w');
for i = 1:total
    if is_db(i)
        rel = strrep(rel_paths{i}, '.jpg', '.png');
        fprintf(fid, 'db\t%s\n', rel);
    else
        fprintf(fid, 'query\t%s\n', rel_paths{i});
    end
end
fclose(fid);

fprintf(1, 'Saved: %s\n', out_mat);
fprintf(1, 'List:  %s\n', out_list);
end

function [db_paths, db_rels, q_paths, q_rels] = read_golden_list(list_path, db_dir, query_dir)
fid = fopen(list_path, 'r');
if fid < 0
    error('Failed to open golden list: %s', list_path);
end
cleanup = onCleanup(@() fclose(fid));
data = textscan(fid, '%s%s', 'Delimiter', '\t');
kinds = data{1};
rels = data{2};
db_paths = {};
db_rels = {};
q_paths = {};
q_rels = {};
for i = 1:numel(kinds)
    kind = kinds{i};
    rel = rels{i};
    if strcmp(kind, 'db')
        db_paths{end + 1, 1} = fullfile(db_dir, rel);
        db_rels{end + 1, 1} = rel;
    elseif strcmp(kind, 'query')
        q_paths{end + 1, 1} = fullfile(query_dir, rel);
        q_rels{end + 1, 1} = rel;
    else
        error('Unexpected entry kind: %s', kind);
    end
end
if isempty(db_paths) || isempty(q_paths)
    error('Golden list missing db/query entries: %s', list_path);
end
end

function root_dir = ensure_root_dir()
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);  % parent of scripts/
local_root = fullfile(repo_root, '247code');
root_dir = local_root;
if ~exist(root_dir, 'dir')
    error('Missing 247code repo: %s (run python scripts/download_assets.py)', root_dir);
end
at_setup_path = fullfile(root_dir, 'at_setup.m');
if ~exist(at_setup_path, 'file')
    error('Missing at_setup.m under: %s', root_dir);
end
end

function out_dir = default_out_dir()
out_dir = fullfile(default_cache_dir(), 'matlab_dump');
end

function cache_dir = default_cache_dir()
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);  % parent of scripts/
cache_dir = fullfile(repo_root, 'assets', 'torii15');
end

function ensure_dir(path)
if ~exist(path, 'dir')
    mkdir(path);
end
end

function opts = parse_opts(opts, varargin)
if mod(numel(varargin), 2) ~= 0
    error('Options must be name-value pairs.');
end
for i = 1:2:numel(varargin)
    name = varargin{i};
    value = varargin{i+1};
    if ~isfield(opts, name)
        error('Unknown option: %s', name);
    end
    opts.(name) = value;
end
end

function img = resize_max_dim(img, max_dim)
if isempty(max_dim) || max_dim <= 0
    return;
end
[h, w] = size(img);
max_hw = max(h, w);
if max_hw <= max_dim
    return;
end
scale = double(max_dim) / double(max_hw);
new_h = max(int32(round(h * scale)), 1);
new_w = max(int32(round(w * scale)), 1);
if exist('imresize', 'file') == 2
    img = imresize(img, [new_h new_w], 'bilinear');
    return;
end
[x, y] = meshgrid(1:w, 1:h);
[xq, yq] = meshgrid(linspace(1, w, double(new_w)), linspace(1, h, double(new_h)));
img_f = single(img);
img_r = interp2(x, y, img_f, xq, yq, 'linear');
if isinteger(img)
    maxv = double(intmax(class(img)));
    img_r = max(min(img_r, maxv), 0);
    img = cast(round(img_r), class(img));
else
    img = cast(img_r, class(img));
end
end
