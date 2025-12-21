% Benchmark MATLAB DenseVLAD encoding speed (breakdown)
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);

addpath(fullfile(repo_root, '247code'));
addpath(fullfile(repo_root, '247code', 'code'));
addpath(fullfile(repo_root, '247code', 'thirdparty', 'yael_matlab_linux64_v438'));
run(fullfile(repo_root, '247code', 'thirdparty', 'vlfeat-0.9.20', 'toolbox', 'vl_setup'));

dictfn = fullfile(repo_root, '247code', 'data', 'dnscnt_RDSIFT_K128.mat');
pcafn = fullfile(repo_root, '247code', 'data', 'dnscnt_RDSIFT_K128_vlad_pcaproj.mat');
db_dir = fullfile(repo_root, 'assets', 'torii15', 'tokyo247', 'database_gsv_vga');

load(pcafn, 'vlad_proj', 'vlad_lambda');
vlad_proj = single(vlad_proj(:,1:4096)');
vlad_wht = single(diag(1./sqrt(vlad_lambda(1:4096))));

% Collect PNG files from nested subdirs
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
        if numel(all_files) >= 100
            break;
        end
    end
    if numel(all_files) >= 100
        break;
    end
end
n = min(100, numel(all_files));

fprintf(1, '>>> MATLAB: Benchmarking %d images (breakdown)...\n', n);

% Load vocabulary once
load(dictfn, 'CX');
CX = bsxfun(@rdivide, CX, sqrt(sum(CX.^2, 1)));
kdtree = vl_kdtreebuild(CX);

t_imread = 0;
t_gray = 0;
t_imdown = 0;
t_phow = 0;
t_rootsift = 0;
t_kdtree = 0;
t_assigns = 0;
t_vl_vlad = 0;
t_pca = 0;

k = size(CX, 2);

for i = 1:n
    imfn = all_files{i};

    tic; img = imread(imfn); t_imread = t_imread + toc;

    tic;
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    t_gray = t_gray + toc;

    tic; img = vl_imdown(img); t_imdown = t_imdown + toc;

    tic; [f, desc] = vl_phow(im2single(img)); t_phow = t_phow + toc;

    tic; desc = relja_rootsift(single(desc)); t_rootsift = t_rootsift + toc;

    % VLAD breakdown
    tic; nn = vl_kdtreequery(kdtree, CX, desc); t_kdtree = t_kdtree + toc;

    tic;
    nd = size(desc, 2);
    assigns = zeros(k, nd, 'single');
    assigns(sub2ind(size(assigns), double(nn), 1:length(nn))) = 1;
    t_assigns = t_assigns + toc;

    tic; vlad = vl_vlad(desc, CX, assigns, 'NormalizeComponents'); t_vl_vlad = t_vl_vlad + toc;

    tic; v = yael_vecs_normalize(vlad_wht * (vlad_proj * vlad)); t_pca = t_pca + toc;
end

fprintf(1, '>>> BREAKDOWN (ms/image):\n');
fprintf(1, '>>>   imread:      %6.1f ms\n', t_imread/n*1000);
fprintf(1, '>>>   rgb2gray:    %6.1f ms\n', t_gray/n*1000);
fprintf(1, '>>>   vl_imdown:   %6.1f ms\n', t_imdown/n*1000);
fprintf(1, '>>>   vl_phow:     %6.1f ms\n', t_phow/n*1000);
fprintf(1, '>>>   rootsift:    %6.1f ms\n', t_rootsift/n*1000);
fprintf(1, '>>>   kdtree_query:%6.1f ms\n', t_kdtree/n*1000);
fprintf(1, '>>>   assigns:     %6.1f ms\n', t_assigns/n*1000);
fprintf(1, '>>>   vl_vlad:     %6.1f ms\n', t_vl_vlad/n*1000);
fprintf(1, '>>>   PCA:         %6.1f ms\n', t_pca/n*1000);
total = t_imread+t_gray+t_imdown+t_phow+t_rootsift+t_kdtree+t_assigns+t_vl_vlad+t_pca;
fprintf(1, '>>>   TOTAL:       %6.1f ms\n', total/n*1000);
