% Benchmark MATLAB DenseVLAD encoding speed
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);

addpath(fullfile(repo_root, '247code'));
addpath(fullfile(repo_root, '247code', 'code'));
run(fullfile(repo_root, '247code', 'thirdparty', 'vlfeat-0.9.20', 'toolbox', 'vl_setup'));

dictfn = fullfile(repo_root, '247code', 'data', 'dnscnt_RDSIFT_K128.mat');
pcafn = fullfile(repo_root, '247code', 'data', 'dnscnt_RDSIFT_K128_vlad_pcaproj.mat');
db_dir = fullfile(repo_root, 'assets', 'torii15', 'tokyo247', 'database_gsv_vga');

load(pcafn, 'vlad_proj', 'vlad_lambda');
vlad_proj = single(vlad_proj(:,1:4096)');
vlad_wht = single(diag(1./sqrt(vlad_lambda(1:4096))));

files = dir(fullfile(db_dir, '**/*.png'));
n = min(100, numel(files));
files = files(1:n);

fprintf(1, '>>> MATLAB: Benchmarking %d images...\n', n);
tic;
for i = 1:n
    imfn = fullfile(files(i).folder, files(i).name);
    vlad = at_image2densevlad(imfn, dictfn);
    v = yael_vecs_normalize(vlad_wht * (vlad_proj * vlad));
end
elapsed = toc;

fprintf(1, '>>> MATLAB: %.1f ms/image (%.2f img/s)\n', elapsed/n*1000, n/elapsed);
