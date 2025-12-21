function bench_phow_breakdown(varargin)
%BENCH_PHOW_BREAKDOWN Detailed PHOW + VLAD timing breakdown (MATLAB).

opts = struct();
opts.reps = 3;
opts.warmup = 1;
opts.max_dim = 640;
opts.use_imdown = true;
opts.imfn = '';
opts.assign_method = 'kdtree';
opts = parse_opts(opts, varargin{:});

root_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '247code');
root_dir = char(java.io.File(root_dir).getCanonicalPath());
orig_dir = pwd;
cd(root_dir);
run('at_setup');
cd(orig_dir);
try
    vl_setnumthreads(1);
catch
end

if isempty(opts.imfn)
    opts.imfn = fullfile(root_dir, 'data', 'example_gsv', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg');
end

dictfn = fullfile(root_dir, 'data', 'dnscnt_RDSIFT_K128.mat');
load(dictfn, 'CX');
CX = bsxfun(@rdivide, CX, sqrt(sum((CX.^2), 1)));
kdtree = vl_kdtreebuild(CX);

sizes = [4 6 8 10];
step = 2;
magnif = 6;
window_size = 1.5;
contrast_threshold = 0.005;
max_size = max(sizes);

% warmup
for w = 1:opts.warmup
    img = imread(opts.imfn);
    if size(img,3) > 1
        img = rgb2gray(img);
    end
    if opts.max_dim > 0
        [h, wimg] = size(img);
        scale = double(opts.max_dim) / double(max(h, wimg));
        if scale < 1
            img = imresize(img, scale, 'bilinear');
        end
    end
    if opts.use_imdown
        img = vl_imdown(img);
    end
    img_single = im2single(img);
    for s = sizes
        off = floor(1.0 + 3.0/2.0 * (max_size - s));
        sigma = s / magnif;
        ims = vl_imsmooth(img_single, sigma);
        [f_s, d_s] = vl_dsift(ims, 'Step', step, 'Size', s, 'Bounds', [off off +inf +inf], ...
            'Norm', 'Fast', 'WindowSize', window_size);
        contrast = f_s(3, :);
        d_s(:, contrast < contrast_threshold) = 0;
    end
end

acc_pre = 0;
acc_ims = zeros(size(sizes));
acc_dsift = zeros(size(sizes));
acc_post = zeros(size(sizes));
acc_concat = 0;
acc_rootsift = 0;
acc_assign = 0;
acc_assign_mat = 0;
acc_vlad = 0;

for r = 1:opts.reps
    t0 = tic;
    img = imread(opts.imfn);
    if size(img,3) > 1
        img = rgb2gray(img);
    end
    if opts.max_dim > 0
        [h, wimg] = size(img);
        scale = double(opts.max_dim) / double(max(h, wimg));
        if scale < 1
            img = imresize(img, scale, 'bilinear');
        end
    end
    if opts.use_imdown
        img = vl_imdown(img);
    end
    img_single = im2single(img);
    acc_pre = acc_pre + toc(t0);

    descs_all = cell(1, numel(sizes));
    for i = 1:numel(sizes)
        s = sizes(i);
        off = floor(1.0 + 3.0/2.0 * (max_size - s));
        sigma = s / magnif;
        t0 = tic;
        ims = vl_imsmooth(img_single, sigma);
        acc_ims(i) = acc_ims(i) + toc(t0);

        t0 = tic;
        [f_s, d_s] = vl_dsift(ims, 'Step', step, 'Size', s, 'Bounds', [off off +inf +inf], ...
            'Norm', 'Fast', 'WindowSize', window_size);
        acc_dsift(i) = acc_dsift(i) + toc(t0);

        t0 = tic;
        contrast = f_s(3, :);
        d_s(:, contrast < contrast_threshold) = 0;
        acc_post(i) = acc_post(i) + toc(t0);
        descs_all{i} = d_s;
    end

    t0 = tic;
    desc = cat(2, descs_all{:});
    acc_concat = acc_concat + toc(t0);

    t0 = tic;
    desc_rs = relja_rootsift(single(desc));
    acc_rootsift = acc_rootsift + toc(t0);

    if strcmpi(opts.assign_method, 'kdtree')
        t0 = tic;
        nn = vl_kdtreequery(kdtree, CX, desc_rs);
        acc_assign = acc_assign + toc(t0);

        t0 = tic;
        assigns = zeros(size(CX, 2), size(desc_rs, 2), 'single');
        assigns(sub2ind(size(assigns), double(nn), 1:length(nn))) = 1;
        acc_assign_mat = acc_assign_mat + toc(t0);
    else
        error('Only assign_method=kdtree is supported in MATLAB benchmark.');
    end

    t0 = tic;
    vlad = vl_vlad(desc_rs, CX, assigns, 'NormalizeComponents');
    acc_vlad = acc_vlad + toc(t0);
end

reps = double(opts.reps);

fprintf(1, 'BENCH matlab\n');
fprintf(1, 'BENCH preprocess %.6f\n', acc_pre / reps);
for i = 1:numel(sizes)
    fprintf(1, 'BENCH imsmooth_%d %.6f\n', sizes(i), acc_ims(i) / reps);
end
for i = 1:numel(sizes)
    fprintf(1, 'BENCH dsift_%d %.6f\n', sizes(i), acc_dsift(i) / reps);
end
for i = 1:numel(sizes)
    fprintf(1, 'BENCH post_%d %.6f\n', sizes(i), acc_post(i) / reps);
end
fprintf(1, 'BENCH phow_concat %.6f\n', acc_concat / reps);
fprintf(1, 'BENCH rootsift %.6f\n', acc_rootsift / reps);
fprintf(1, 'BENCH assign %.6f\n', acc_assign / reps);
fprintf(1, 'BENCH assign_mat %.6f\n', acc_assign_mat / reps);
fprintf(1, 'BENCH vlad %.6f\n', acc_vlad / reps);

end

function opts = parse_opts(opts, varargin)
if isempty(varargin)
    return
end
if mod(numel(varargin), 2) ~= 0
    error('Options must be name/value pairs.')
end
for i = 1:2:numel(varargin)
    key = varargin{i};
    val = varargin{i+1};
    if ~ischar(key)
        error('Option keys must be strings.')
    end
    key = lower(key);
    switch key
        case 'reps'
            opts.reps = val;
        case 'warmup'
            opts.warmup = val;
        case 'max_dim'
            opts.max_dim = val;
        case 'use_imdown'
            opts.use_imdown = logical(val);
        case 'imfn'
            opts.imfn = val;
        case 'assign_method'
            opts.assign_method = val;
        otherwise
            error('Unknown option: %s', key);
    end
end
end
