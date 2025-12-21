% Regenerate example VLAD files to ensure parity with current MATLAB code
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);

addpath(fullfile(repo_root, '247code'));
addpath(fullfile(repo_root, '247code', 'code'));
addpath(fullfile(repo_root, '247code', 'thirdparty', 'yael_matlab_linux64_v438'));
run(fullfile(repo_root, '247code', 'thirdparty', 'vlfeat-0.9.20', 'toolbox', 'vl_setup'));

dictfn = fullfile(repo_root, '247code', 'data', 'dnscnt_RDSIFT_K128.mat');
pcafn = fullfile(repo_root, '247code', 'data', 'dnscnt_RDSIFT_K128_vlad_pcaproj.mat');
example_dir = fullfile(repo_root, '247code', 'data', 'example_gsv');

% Load PCA projection
load(pcafn, 'vlad_proj', 'vlad_lambda');
vlad_proj = single(vlad_proj(:,1:4096)');
vlad_wht = single(diag(1./sqrt(vlad_lambda(1:4096))));

% Find example images
imgs = dir(fullfile(example_dir, '*.jpg'));
fprintf(1, '>>> Found %d example images\n', numel(imgs));

for i = 1:numel(imgs)
    imfn = fullfile(example_dir, imgs(i).name);
    [~, basename, ~] = fileparts(imgs(i).name);
    outfn = fullfile(example_dir, [basename '.dict_grid.dnsvlad.mat']);

    fprintf(1, '>>> Processing %s...\n', basename);

    % Compute raw VLAD using original function
    vlad_raw = at_image2densevlad(imfn, dictfn);

    % Apply PCA whitening
    v = yael_vecs_normalize(vlad_wht * (vlad_proj * vlad_raw));

    % Save in same format - use 'vlad' key for test compatibility
    vlad = v;
    save(outfn, 'vlad', '-v7');

    fprintf(1, '>>>   Saved: %s\n', outfn);
end

fprintf(1, '>>> Done regenerating example VLADs\n');
