function dump_tokyo247_golden(varargin)
%DUMP_TOKYO247_GOLDEN Generate DenseVLAD golden references for Tokyo 24/7.
%  dump_tokyo247_golden() writes a v7.3 .mat plus a text list to:
%    ~/Library/Caches/dvlad/torii15/matlab_dump/
%
%  Name-value options:
%    'seed'   : RNG seed (default 1337)
%    'num_db' : number of DB images (default 5)
%    'num_q'  : number of query images (default 5)
%    'out_dir': output directory (default matlab_dump cache)

opts = struct();
opts.seed = 1337;
opts.num_db = 5;
opts.num_q = 5;
opts.out_dir = '';
opts = parse_opts(opts, varargin{:});

if isempty(opts.out_dir)
    opts.out_dir = fullfile(getenv('HOME'), 'Library', 'Caches', 'dvlad', 'torii15', 'matlab_dump');
end
if ~exist(opts.out_dir, 'dir')
    mkdir(opts.out_dir);
end

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

cache_dir = fullfile(getenv('HOME'), 'Library', 'Caches', 'dvlad', 'torii15');
root_dir = fullfile(cache_dir, '247code');
tokyo_root = fullfile(cache_dir, 'tokyo247');
db_dir = fullfile(tokyo_root, 'database_gsv_vga');
query_dir = fullfile(cache_dir, '247query_subset_v2', '247query_subset_v2');
dbstruct_path = fullfile(tokyo_root, 'tokyo247.mat');

addpath(root_dir);
addpath(fullfile(root_dir, 'code'));
run(fullfile(root_dir, 'thirdparty', 'vlfeat-0.9.20', 'toolbox', 'vl_setup'));

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
    img = vl_imdown(img);
    img_single = single(img);
    if isinteger(img)
        img_single = img_single ./ single(intmax(class(img)));
    end
    [~, desc] = vl_phow(img_single);
    desc = relja_rootsift(single(desc));
    vlad = relja_computeVLAD(desc, CX, kdtree);
    vlad_pre(:, i) = single(vlad);
    v = vlad_wht * (vlad_proj * vlad);
    v = v / norm(v);
    vlad_4096(:, i) = single(v);
    fprintf(1, '  %d/%d\n', i, total);
end

out_mat = fullfile(opts.out_dir, 'tokyo247_golden.mat');
out_list = fullfile(opts.out_dir, 'tokyo247_golden_list.txt');
save(out_mat, 'vlad_pre', 'vlad_4096', 'is_db', 'rel_paths', 'opts', '-v7.3');

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
