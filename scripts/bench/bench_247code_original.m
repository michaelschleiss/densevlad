repo = '/home/administrator/projects/densevlad';
image_path = fullfile(repo, '247code', 'data', 'example_gsv', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg');
dictfn = fullfile(repo, '247code', 'data', 'dnscnt_RDSIFT_K128.mat');
shipped_path = fullfile(repo, '247code', 'data', 'example_gsv', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_000.dict_grid.dnsvlad.mat');

cd(fullfile(repo, '247code'));
run('at_setup');
run(fullfile(repo, '247code', 'thirdparty', 'vlfeat-0.9.20', 'toolbox', 'vl_setup'));
fprintf(1, 'vl_version: %s\n', vl_version);
fprintf(1, 'vl_phow: %s\n', which('vl_phow'));
fprintf(1, 'vl_dsift: %s\n', which('vl_dsift'));
load(dictfn, 'CX');
CX = at_l2normalize_col(CX);
kdtree = vl_kdtreebuild(CX);
zero_desc = zeros(size(CX,1), 1, class(CX));
cached_zero_tie = vl_kdtreequery(kdtree, CX, zero_desc);

if exist('vl_threads', 'file')
  vl_threads(1);
end
if exist('vl_setnumthreads', 'file')
  vl_setnumthreads(1);
end

reps = 3;
warmup = 1;
default_threads = 16;
sizes = [4 6 8 10];
step = 2;
magnif = 6;
window_size = 1.5;
contrast_threshold = 0.005;
max_size = max(sizes);

% Try to align MATLAB BLAS threads with VLFeat threads.
setenv('OMP_NUM_THREADS', num2str(default_threads));
setenv('MKL_NUM_THREADS', num2str(default_threads));
setenv('OPENBLAS_NUM_THREADS', num2str(default_threads));

if exist('vl_threads', 'file')
  vl_threads(default_threads);
end
if exist('vl_setnumthreads', 'file')
  vl_setnumthreads(default_threads);
end

% Warmup loop (pre-resize + DenseVLAD)
for wi = 1:warmup
  img = imread(image_path);
  if (size(img,3)>1)
    img=rgb2gray(img);
  end
  img = imresize(img, [480 640], 'bilinear');
  img = vl_imdown(img);
  img_single = im2single(img);
  [f, desc] = vl_phow(img_single);
  desc = relja_rootsift(single(desc));
  v = relja_computeVLAD(desc, CX, kdtree);
end

acc_noresize = 0;
acc_noresize_preprocess = 0;
acc_noresize_phow = 0;
acc_noresize_rootsift = 0;
acc_noresize_vlad = 0;
v_noresize = [];

% Timing loop for image 012_000 (original path: no pre-resize)
for r = 1:reps
  t0 = tic;
  img = imread(image_path);
  if (size(img,3)>1)
    img=rgb2gray(img);
  end
  img = vl_imdown(img);
  t_pre = toc(t0);
  acc_noresize_preprocess = acc_noresize_preprocess + t_pre;

  t0 = tic;
  [f, desc] = vl_phow(im2single(img));
  t_phow = toc(t0);
  acc_noresize_phow = acc_noresize_phow + t_phow;

  t0 = tic;
  desc = relja_rootsift(single(desc));
  t_rootsift = toc(t0);
  acc_noresize_rootsift = acc_noresize_rootsift + t_rootsift;

  t0 = tic;
  v = relja_computeVLAD(desc, CX, kdtree);
  t_vlad = toc(t0);
  acc_noresize_vlad = acc_noresize_vlad + t_vlad;
  v_noresize = v;
  acc_noresize = acc_noresize + t_pre + t_phow + t_rootsift + t_vlad;
end

acc_resize = 0;
acc_resize_preprocess = 0;
acc_resize_phow = 0;
acc_resize_rootsift = 0;
acc_resize_vlad = 0;
v_resize = [];
% Timing loop for image 012_000 (pre-resize + DenseVLAD)
for r = 1:reps
  t0 = tic;
  img = imread(image_path);
  if (size(img,3)>1)
    img=rgb2gray(img);
  end
  img = imresize(img, [480 640], 'bilinear');
  img = vl_imdown(img);
  t_pre = toc(t0);
  acc_resize_preprocess = acc_resize_preprocess + t_pre;

  t0 = tic;
  [f, desc] = vl_phow(im2single(img));
  t_phow = toc(t0);
  acc_resize_phow = acc_resize_phow + t_phow;

  t0 = tic;
  desc = relja_rootsift(single(desc));
  t_rootsift = toc(t0);
  acc_resize_rootsift = acc_resize_rootsift + t_rootsift;

  t0 = tic;
  v = relja_computeVLAD(desc, CX, kdtree);
  t_vlad = toc(t0);
  acc_resize_vlad = acc_resize_vlad + t_vlad;
  v_resize = v;
  acc_resize = acc_resize + t_pre + t_phow + t_rootsift + t_vlad;
end

fprintf('BENCH matlab_original_at_image2densevlad noresize=%.6f preresize=%.6f\n', acc_noresize / reps, acc_resize / reps);
fprintf('BENCH matlab_preprocess             noresize=%.6f preresize=%.6f\n', acc_noresize_preprocess / reps, acc_resize_preprocess / reps);
fprintf('BENCH matlab_phow                   noresize=%.6f preresize=%.6f\n', acc_noresize_phow / reps, acc_resize_phow / reps);
fprintf('BENCH matlab_rootsift               noresize=%.6f preresize=%.6f\n', acc_noresize_rootsift / reps, acc_resize_rootsift / reps);
fprintf('BENCH matlab_computeVLAD            noresize=%.6f preresize=%.6f\n', acc_noresize_vlad / reps, acc_resize_vlad / reps);

% Parity check (cosine similarity) vs shipped asset.
load(shipped_path, 'vlad');
vlad = vlad(:);
assert(isa(vlad, 'single'), 'Expected single-precision VLADs.');

% KDTREE path (original)
v = v_noresize(:);
assert(numel(v) == numel(vlad), 'Shipped VLAD size mismatch (kdtree).');
a = double(v);
b = double(vlad);
denom = norm(a) * norm(b);
cos_sim = NaN;
if denom > 0
  cos_sim = dot(a, b) / denom;
end
fprintf('BENCH matlab_cosine_vs_shipped_kdtree %.9f\n', cos_sim);

% Enforce parity against shipped using the best path found.
best_cos = cos_sim;
assert(best_cos >= 0.999, 'Shipped VLAD cosine similarity %.6f < 0.999.', best_cos);
