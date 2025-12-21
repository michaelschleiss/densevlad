function eval_tokyo247_densevlad(varargin)
%EVAL_TOKYO247_DENSEVLAD DenseVLAD baseline evaluation on Tokyo 24/7.
%   This script computes DenseVLAD descriptors (4096-D) for database and
%   query images, then reports recall@N for all/day/night queries.
%
%   Name-value options:
%     'dbstruct_path' : path to tokyo247.mat (dbStruct)
%     'db_dir'        : root dir containing 03814/...
%     'query_dir'     : directory containing *.jpg + *.csv queries
%     'out_dir'       : cache dir for descriptors
%     'limit_db'      : limit number of DB images (0 = full)
%     'limit_q'       : limit number of queries (0 = full)
%     'force'         : true to recompute caches
%     'recall_n'      : vector of N values for recall

opts = struct();
opts.dbstruct_path = '';
opts.db_dir = '';
opts.query_dir = '';
opts.out_dir = '';
opts.limit_db = 0;
opts.limit_q = 0;
opts.force = false;
opts.recall_n = [1 5 10 20];
opts.max_dim = 640;
opts.use_imdown = false;

opts = parse_opts(opts, varargin{:});

home = getenv('HOME');
cache_dir = fullfile(home, 'Library', 'Caches', 'densevlad', 'torii15');
root_dir = fullfile(cache_dir, 'tokyo247');
if isempty(opts.dbstruct_path)
    opts.dbstruct_path = fullfile(root_dir, 'tokyo247.mat');
end
if isempty(opts.db_dir)
    opts.db_dir = fullfile(root_dir, 'database_gsv_vga');
end
if isempty(opts.query_dir)
    opts.query_dir = fullfile(cache_dir, '247query_subset_v2');
end
if isempty(opts.out_dir)
    opts.out_dir = fullfile(root_dir, 'matlab_densevlad_cache');
end

fprintf(1, 'dbstruct: %s\n', opts.dbstruct_path);
fprintf(1, 'db_dir:   %s\n', opts.db_dir);
fprintf(1, 'query:    %s\n', opts.query_dir);
fprintf(1, 'out_dir:  %s\n', opts.out_dir);

if ~exist(opts.out_dir, 'dir')
    mkdir(opts.out_dir);
end

code_root = fullfile(cache_dir, '247code');
addpath(code_root);
addpath(fullfile(code_root, 'code'));
addpath(fileparts(mfilename('fullpath')));
run(fullfile(code_root, 'thirdparty', 'vlfeat-0.9.20', 'toolbox', 'vl_setup'));

dictfn = fullfile(code_root, 'data', 'dnscnt_RDSIFT_K128.mat');
pcafn = fullfile(code_root, 'data', 'dnscnt_RDSIFT_K128_vlad_pcaproj.mat');

fprintf(1, 'Loading vocabulary...\n');
load(dictfn, 'CX');
CX = bsxfun(@rdivide, CX, sqrt(sum(CX.^2, 1)));
kdtree = vl_kdtreebuild(CX);

fprintf(1, 'Loading PCA...\n');
load(pcafn, 'vlad_proj', 'vlad_lambda');
vladdim = 4096;
vlad_proj = single(vlad_proj(:, 1:vladdim)');
vlad_wht = single(diag(1 ./ sqrt(vlad_lambda(1:vladdim))));

fprintf(1, 'Loading dbStruct...\n');
dbs = load(opts.dbstruct_path, 'dbStruct');
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
utm_db = dbStruct.utmDb;
utm_q = dbStruct.utmQ;
pos_dist_thr = double(dbStruct.posDistThr);

num_db = numel(db_images);
num_q = numel(q_images);

if opts.limit_db > 0
    num_db = min(num_db, opts.limit_db);
    db_images = db_images(1:num_db);
    utm_db = utm_db(:, 1:num_db);
end
if opts.limit_q > 0
    num_q = min(num_q, opts.limit_q);
    q_images = q_images(1:num_q);
    utm_q = utm_q(:, 1:num_q);
end

db_mat = fullfile(opts.out_dir, 'densevlad_4096_db.mat');
q_mat = fullfile(opts.out_dir, 'densevlad_4096_q.mat');

[db_file, db_done] = open_desc_file(db_mat, vladdim, num_db, 'db_desc', 'db_done', opts.force);
[q_file, q_done] = open_desc_file(q_mat, vladdim, num_q, 'q_desc', 'q_done', opts.force);

fprintf(1, 'Computing DB descriptors (%d images)...\n', num_db);
db_done = compute_descs(db_file, 'db_desc', db_done, db_images, opts.db_dir, true, opts.max_dim, opts.use_imdown, CX, kdtree, vlad_proj, vlad_wht);
db_file.db_done = db_done;

fprintf(1, 'Computing query descriptors (%d images)...\n', num_q);
q_done = compute_descs(q_file, 'q_desc', q_done, q_images, opts.query_dir, false, opts.max_dim, opts.use_imdown, CX, kdtree, vlad_proj, vlad_wht);
q_file.q_done = q_done;

fprintf(1, 'Loading descriptors for evaluation...\n');
db_desc = db_file.db_desc;
q_desc = q_file.q_desc;

fprintf(1, 'Computing positives (radius %.2f m)...\n', pos_dist_thr);
positives = cell(1, num_q);
for i = 1:num_q
    d = utm_db - utm_q(:, i);
    dist = sqrt(sum(d .* d, 1));
    positives{i} = find(dist <= pos_dist_thr);
end

fprintf(1, 'Computing similarity matrix...\n');
scores = db_desc' * q_desc;

fprintf(1, 'Computing recall@N...\n');
time_labels = load_time_labels(opts.query_dir, q_images);
mask_day = strcmp(time_labels, 'D');
mask_night = strcmp(time_labels, 'S') | strcmp(time_labels, 'N');

recall_all = compute_recall(scores, positives, opts.recall_n, true(1, num_q));
recall_day = compute_recall(scores, positives, opts.recall_n, mask_day);
recall_night = compute_recall(scores, positives, opts.recall_n, mask_night);

fprintf(1, 'Recall@N (all):\n');
disp(recall_all);
fprintf(1, 'Recall@N (day):\n');
disp(recall_day);
fprintf(1, 'Recall@N (sunset+night):\n');
disp(recall_night);
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

function [mf, done] = open_desc_file(path, dim, count, desc_name, done_name, force)
if force && exist(path, 'file')
    delete(path);
end

mf = matfile(path, 'Writable', true);
vars = whos(mf);
varnames = {vars.name};

if ~any(strcmp(varnames, desc_name))
    mf.(desc_name) = zeros(dim, count, 'single');
end
if ~any(strcmp(varnames, done_name))
    mf.(done_name) = false(1, count);
end
done = mf.(done_name);
end

function done = compute_descs(mf, desc_name, done, images, root_dir, use_png, max_dim, use_imdown, CX, kdtree, vlad_proj, vlad_wht)
count = numel(images);
for i = 1:count
    if done(i)
        continue;
    end
    rel = images{i};
    if use_png && endsWith(rel, '.jpg')
        rel = strrep(rel, '.jpg', '.png');
    end
    imfn = fullfile(root_dir, rel);
    v = compute_densevlad(imfn, max_dim, use_imdown, CX, kdtree, vlad_proj, vlad_wht);
    mf.(desc_name)(:, i) = v;
    done(i) = true;
    if mod(i, 50) == 0 || i == count
        fprintf(1, '  %d/%d\n', i, count);
    end
end
end

function v = compute_densevlad(imfn, max_dim, use_imdown, CX, kdtree, vlad_proj, vlad_wht)
img = imread(imfn);
if size(img, 3) > 1
    img = rgb2gray(img);
end
img = resize_max_dim(img, max_dim);
if use_imdown
    img = vl_imdown(img);
end
img_single = single(img);
if isinteger(img)
    img_single = img_single ./ single(intmax(class(img)));
end
[~, desc] = vl_phow(img_single);
desc = relja_rootsift(single(desc));
vlad = relja_computeVLAD(desc, CX, kdtree);
v = vlad_proj * vlad;
v = vlad_wht * v;
v = v / norm(v);
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

function labels = load_time_labels(query_dir, q_images)
num_q = numel(q_images);
labels = cell(1, num_q);
for i = 1:num_q
    qname = q_images{i};
    csv_fn = fullfile(query_dir, strrep(qname, '.jpg', '.csv'));
    fid = fopen(csv_fn, 'r');
    line = fgetl(fid);
    fclose(fid);
    parts = strsplit(line, ',');
    labels{i} = parts{6};
end
end

function rec = compute_recall(scores, positives, ns, mask)
num_q = numel(positives);
ns = ns(:)';
rec = zeros(1, numel(ns));
valid = find(mask);
for nidx = 1:numel(ns)
    n = ns(nidx);
    hits = 0;
    for qi = valid
        pos = positives{qi};
        if isempty(pos)
            continue;
        end
        [~, idx] = maxk(scores(:, qi), n);
        if any(ismember(idx, pos))
            hits = hits + 1;
        end
    end
    rec(nidx) = hits / numel(valid);
end
end
