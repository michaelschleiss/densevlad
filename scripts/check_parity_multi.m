% Check optimization parity across multiple images
% Validates that optimized VLAD produces equivalent results to vl_vlad

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, '247code'));
addpath(fullfile(repo_root, '247code', 'code'));
addpath(fullfile(repo_root, '247code', 'thirdparty', 'yael_matlab_linux64_v438'));
run(fullfile(repo_root, '247code', 'thirdparty', 'vlfeat-0.9.20', 'toolbox', 'vl_setup'));

load(fullfile(repo_root, '247code', 'data', 'dnscnt_RDSIFT_K128.mat'), 'CX');
CX = single(bsxfun(@rdivide, CX, sqrt(sum(CX.^2, 1))));
k = size(CX, 2); D = size(CX, 1);

% Collect 20 test images
db_dir = fullfile(repo_root, 'assets', 'torii15', 'tokyo247', 'database_gsv_vga');
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
n = min(20, numel(all_files));

fprintf(1, '>>> Testing %d images\n', n);

max_diffs = zeros(1, n);
mean_diffs = zeros(1, n);
pct_exact = zeros(1, n);

for i = 1:n
    img = imread(all_files{i});
    if size(img, 3) > 1, img = rgb2gray(img); end
    img = vl_imdown(img);
    [~, desc] = vl_phow(im2single(img));
    desc = relja_rootsift(single(desc));
    [~, nn] = max(CX' * desc, [], 1);  % matmul NN
    nd = size(desc, 2);

    % Reference: vl_vlad
    assigns = zeros(k, nd, 'single');
    assigns(sub2ind(size(assigns), double(nn), 1:nd)) = 1;
    vlad_ref = vl_vlad(desc, CX, assigns, 'NormalizeComponents');

    % Optimized: vectorized
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

    diff = abs(vlad_ref - vlad_opt);
    max_diffs(i) = max(diff);
    mean_diffs(i) = mean(diff);
    pct_exact(i) = 100 * mean(vlad_ref == vlad_opt);
end

fprintf(1, '>>> Results across %d images:\n', n);
fprintf(1, '>>>   max diff:  min=%.2e, max=%.2e, mean=%.2e\n', min(max_diffs), max(max_diffs), mean(max_diffs));
fprintf(1, '>>>   mean diff: min=%.2e, max=%.2e, mean=%.2e\n', min(mean_diffs), max(mean_diffs), mean(mean_diffs));
fprintf(1, '>>>   bit-exact: min=%.1f%%, max=%.1f%%, mean=%.1f%%\n', min(pct_exact), max(pct_exact), mean(pct_exact));
if max(max_diffs) < 1e-6
    fprintf(1, '>>>   all within 1e-6: YES\n');
else
    fprintf(1, '>>>   all within 1e-6: NO (max=%.2e)\n', max(max_diffs));
end
fprintf(1, '>>> DONE\n');
