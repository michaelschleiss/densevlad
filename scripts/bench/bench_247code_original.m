repo = '/home/administrator/projects/densevlad';
image_path = fullfile(repo, '247code', 'data', 'example_gsv', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_000.jpg');
dictfn = fullfile(repo, '247code', 'data', 'dnscnt_RDSIFT_K128.mat');
shipped_path = fullfile(repo, '247code', 'data', 'example_gsv', 'L-NLvGeZ6JHX6JO8Xnf_BA_012_000.dict_grid.dnsvlad.mat');

cd(fullfile(repo, '247code'));
run('at_setup');
load(dictfn, 'CX');
CX = at_l2normalize_col(CX);
kdtree = vl_kdtreebuild(CX);

reps = 3;
warmup = 1;
sizes = [4 6 8 10];
step = 2;
magnif = 6;
window_size = 1.5;
contrast_threshold = 0.005;
max_size = max(sizes);

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

% Total runtime vs vl_threads sweep (uses no-resize path)
if exist('vl_threads', 'file')
  sweep_threads = [1 8 16 32];
  for ti = 1:numel(sweep_threads)
    vl_threads(sweep_threads(ti));
    acc_total = 0;
    for r = 1:reps
      t0 = tic;
      img = imread(image_path);
      if (size(img,3)>1)
        img=rgb2gray(img);
      end
      img = vl_imdown(img);
      img_single = im2single(img);
      [f, desc] = vl_phow(img_single);
      desc = relja_rootsift(single(desc));
      v = relja_computeVLAD(desc, CX, kdtree);
      acc_total = acc_total + toc(t0);
    end
    fprintf('BENCH matlab_total_threads %d %.6f\n', sweep_threads(ti), acc_total / reps);
  end
else
  disp('vl_threads not found');
end
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
  acc_resize = acc_resize + t_pre + t_phow + t_rootsift + t_vlad;
end

fprintf('BENCH matlab_original_at_image2densevlad noresize=%.6f preresize=%.6f\n', acc_noresize / reps, acc_resize / reps);
fprintf('BENCH matlab_preprocess             noresize=%.6f preresize=%.6f\n', acc_noresize_preprocess / reps, acc_resize_preprocess / reps);
fprintf('BENCH matlab_phow                   noresize=%.6f preresize=%.6f\n', acc_noresize_phow / reps, acc_resize_phow / reps);
fprintf('BENCH matlab_rootsift               noresize=%.6f preresize=%.6f\n', acc_noresize_rootsift / reps, acc_resize_rootsift / reps);
fprintf('BENCH matlab_computeVLAD            noresize=%.6f preresize=%.6f\n', acc_noresize_vlad / reps, acc_resize_vlad / reps);

% Parity check (bit-identical) for original path vs shipped asset.
v = v_noresize;
load(shipped_path, 'vlad');
v = v(:);
vlad = vlad(:);
assert(numel(v) == numel(vlad), 'Shipped VLAD size mismatch.');
assert(isa(v, 'single') && isa(vlad, 'single'), 'Expected single-precision VLADs.');
if ~isequal(v, vlad)
  idx = find(v ~= vlad, 1, 'first');
  error('Shipped VLAD mismatch at index %d (got %.9g, expected %.9g).', idx, v(idx), vlad(idx));
end
fprintf('BENCH matlab_original_bit_identical 1\n');
