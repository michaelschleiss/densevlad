% Benchmark DenseVLAD optimization approaches
% Compare: kdtree vs matmul for NN, dense vs sparse for VLAD

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);

addpath(fullfile(repo_root, '247code'));
addpath(fullfile(repo_root, '247code', 'code'));
addpath(fullfile(repo_root, '247code', 'thirdparty', 'yael_matlab_linux64_v438'));
run(fullfile(repo_root, '247code', 'thirdparty', 'vlfeat-0.9.20', 'toolbox', 'vl_setup'));

dictfn = fullfile(repo_root, '247code', 'data', 'dnscnt_RDSIFT_K128.mat');
db_dir = fullfile(repo_root, 'assets', 'torii15', 'tokyo247', 'database_gsv_vga');

% Load vocabulary
load(dictfn, 'CX');
CX = bsxfun(@rdivide, CX, sqrt(sum(CX.^2, 1)));  % L2 normalize
CX = single(CX);
k = size(CX, 2);
D = size(CX, 1);

% Build kdtree once
kdtree = vl_kdtreebuild(CX);

% Collect test images
all_files = {};
level1 = dir(db_dir);
level1 = level1([level1.isdir] & ~ismember({level1.name}, {'.', '..'}));
for i = 1:numel(level1)
    l1_path = fullfile(db_dir, level1(i).name);
    level2 = dir(l1_path);
    level2 = level2([level2.isdir] & ~ismember({level2.name}, {'.', '..'}));
    for j = 1:numel(level2)
        l2_path = fullfile(l1_path, level2(j).name);
        pngs = dir(fullfile(l2_path, '*.png'));
        for p = 1:numel(pngs)
            all_files{end+1} = fullfile(l2_path, pngs(p).name);
        end
        if numel(all_files) >= 20
            break;
        end
    end
    if numel(all_files) >= 20
        break;
    end
end
n_images = min(20, numel(all_files));

fprintf(1, '>>> Benchmarking %d images\n\n', n_images);

% Pre-extract descriptors for fair comparison
fprintf(1, '>>> Extracting descriptors...\n');
all_descs = cell(1, n_images);
for i = 1:n_images
    img = imread(all_files{i});
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    img = vl_imdown(img);
    [~, desc] = vl_phow(im2single(img));
    all_descs{i} = relja_rootsift(single(desc));
end
fprintf(1, '>>> Done. Average descriptors per image: %d\n\n', ...
    round(mean(cellfun(@(x) size(x,2), all_descs))));

%% ========== TEST 1: NN Assignment Methods ==========
fprintf(1, '========== NN ASSIGNMENT METHODS ==========\n');

n_runs = 3;

% Method 1: vl_kdtreequery (original)
t_kdtree = zeros(1, n_runs);
for r = 1:n_runs
    tic;
    for i = 1:n_images
        nn = vl_kdtreequery(kdtree, CX, all_descs{i});
    end
    t_kdtree(r) = toc;
end
fprintf(1, '>>> kdtree:     %6.1f ms/image (std: %.1f)\n', ...
    mean(t_kdtree)/n_images*1000, std(t_kdtree)/n_images*1000);

% Method 2: Exhaustive matmul (float32)
t_matmul32 = zeros(1, n_runs);
for r = 1:n_runs
    tic;
    for i = 1:n_images
        desc = all_descs{i};
        Gram = CX' * desc;
        [~, nn] = max(Gram, [], 1);
    end
    t_matmul32(r) = toc;
end
fprintf(1, '>>> matmul32:   %6.1f ms/image (std: %.1f)\n', ...
    mean(t_matmul32)/n_images*1000, std(t_matmul32)/n_images*1000);

% Method 3: Explicit L2 distance (for non-normalized centers)
t_l2dist = zeros(1, n_runs);
CX_sq = sum(CX.^2, 1);  % precompute
for r = 1:n_runs
    tic;
    for i = 1:n_images
        desc = all_descs{i};
        desc_sq = sum(desc.^2, 1);
        D_mat = bsxfun(@plus, CX_sq', desc_sq) - 2 * (CX' * desc);
        [~, nn] = min(D_mat, [], 1);
    end
    t_l2dist(r) = toc;
end
fprintf(1, '>>> L2 dist:    %6.1f ms/image (std: %.1f)\n', ...
    mean(t_l2dist)/n_images*1000, std(t_l2dist)/n_images*1000);

% Verify equivalence
desc_test = all_descs{1};
nn_kdtree = vl_kdtreequery(kdtree, CX, desc_test);
[~, nn_matmul] = max(CX' * desc_test, [], 1);
D_l2 = bsxfun(@plus, CX_sq', sum(desc_test.^2,1)) - 2*(CX'*desc_test);
[~, nn_l2] = min(D_l2, [], 1);
fprintf(1, '>>> Equivalence: kdtree==matmul: %.1f%%, kdtree==L2: %.1f%%\n', ...
    100*mean(nn_kdtree == nn_matmul), 100*mean(nn_kdtree == nn_l2));

% Investigate mismatches - are they due to ties?
mismatches = find(nn_kdtree ~= nn_matmul);
if ~isempty(mismatches)
    fprintf(1, '>>> Investigating %d mismatches (first 5)...\n', numel(mismatches));
    for j = 1:min(5, numel(mismatches))
        idx = mismatches(j);
        d_kd = desc_test(:, idx);
        dist_kd = norm(CX(:, nn_kdtree(idx)) - d_kd);
        dist_mm = norm(CX(:, nn_matmul(idx)) - d_kd);
        fprintf(1, '>>>   desc %d: kdtree->%d (d=%.6f), matmul->%d (d=%.6f), diff=%.2e\n', ...
            idx, nn_kdtree(idx), dist_kd, nn_matmul(idx), dist_mm, abs(dist_kd-dist_mm));
    end
end

%% ========== TEST 2: VLAD Accumulation Methods ==========
fprintf(1, '\n========== VLAD ACCUMULATION METHODS ==========\n');

% Get assignments using matmul (faster)
nn_all = cell(1, n_images);
for i = 1:n_images
    [~, nn_all{i}] = max(CX' * all_descs{i}, [], 1);
end

% Method 1: Dense assignment matrix (original vl_vlad approach)
t_dense = zeros(1, n_runs);
for r = 1:n_runs
    tic;
    for i = 1:n_images
        desc = all_descs{i};
        nn = nn_all{i};
        nd = size(desc, 2);
        assigns = zeros(k, nd, 'single');
        assigns(sub2ind(size(assigns), double(nn), 1:nd)) = 1;
        vlad = vl_vlad(desc, CX, assigns, 'NormalizeComponents');
    end
    t_dense(r) = toc;
end
fprintf(1, '>>> dense+vl_vlad:    %6.1f ms/image (std: %.1f)\n', ...
    mean(t_dense)/n_images*1000, std(t_dense)/n_images*1000);

% Method 2: Vectorized VLAD (no vl_vlad)
t_vectorized = zeros(1, n_runs);
for r = 1:n_runs
    tic;
    for i = 1:n_images
        desc = all_descs{i};
        nn = nn_all{i};
        nd = size(desc, 2);

        enc = zeros(D, k, 'single');
        for c = 1:k
            mask = (nn == c);
            if any(mask)
                enc(:, c) = sum(desc(:, mask), 2) - sum(mask) * CX(:, c);
            end
        end
        % Intra-norm + global norm
        norms = sqrt(sum(enc.^2, 1));
        norms(norms == 0) = 1;
        enc = bsxfun(@rdivide, enc, norms);
        vlad = enc(:) / norm(enc(:));
    end
    t_vectorized(r) = toc;
end
fprintf(1, '>>> vectorized VLAD:  %6.1f ms/image (std: %.1f)\n', ...
    mean(t_vectorized)/n_images*1000, std(t_vectorized)/n_images*1000);

% Verify VLAD equivalence
desc = all_descs{1}; nn = nn_all{1}; nd = size(desc, 2);
assigns = zeros(k, nd, 'single');
assigns(sub2ind(size(assigns), double(nn), 1:nd)) = 1;
vlad_ref = vl_vlad(desc, CX, assigns, 'NormalizeComponents');

enc = zeros(D, k, 'single');
for c = 1:k
    mask = (nn == c);
    if any(mask)
        enc(:, c) = sum(desc(:, mask), 2) - sum(mask) * CX(:, c);
    end
end
norms = sqrt(sum(enc.^2, 1));
norms(norms == 0) = 1;
enc = bsxfun(@rdivide, enc, norms);
vlad_opt = enc(:) / norm(enc(:));

max_diff = max(abs(vlad_ref - vlad_opt));
if max_diff < 1e-6
    fprintf(1, '>>> VLAD max diff: %.2e (within float32: YES)\n', max_diff);
else
    fprintf(1, '>>> VLAD max diff: %.2e (within float32: NO)\n', max_diff);
end

%% ========== SUMMARY ==========
fprintf(1, '\n========== SUMMARY ==========\n');
fprintf(1, '>>> Best NN: matmul32 (%.1fx faster than kdtree)\n', mean(t_kdtree)/mean(t_matmul32));
fprintf(1, '>>> Best VLAD: vectorized (%.1fx faster than vl_vlad)\n', mean(t_dense)/mean(t_vectorized));
fprintf(1, '>>> Total speedup: %.1fx\n', (mean(t_kdtree)+mean(t_dense))/(mean(t_matmul32)+mean(t_vectorized)));
fprintf(1, '>>> DONE\n');
