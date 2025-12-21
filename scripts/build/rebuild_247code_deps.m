function rebuild_247code_deps()
%REBUILD_247CODE_DEPS Rebuild 247code third-party deps and verify MATLAB paths.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
code_root = fullfile(repo_root, '247code');
yael_root = fullfile(code_root, 'thirdparty', 'yael_matlab_linux64_v438');
vlfeat_root = fullfile(code_root, 'thirdparty', 'vlfeat-0.9.20');

if ~exist(code_root, 'dir')
    error('Expected 247code directory not found: %s', code_root);
end

fprintf(1, 'Rebuilding Yael MEX in %s\n', yael_root);
if ~exist(yael_root, 'dir')
    error('Yael directory missing: %s', yael_root);
end
yael_src = fullfile(yael_root, '..', 'yael');
if exist(yael_src, 'dir') ~= 7
    fprintf(1, 'Yael source tree not found at %s; skipping rebuild.\n', yael_src);
else
    old_yael = pwd;
    cleanup_yael = onCleanup(@() cd(old_yael));
    cd(yael_root);
    run('Make.m');
end

fprintf(1, 'Rebuilding VLFeat in %s\n', vlfeat_root);
if ~exist(vlfeat_root, 'dir')
    error('VLFeat directory missing: %s', vlfeat_root);
end
old = pwd;
cleanup = onCleanup(@() cd(old));
cd(vlfeat_root);
system('make clean');
status = system('make DISABLE_OPENMP=yes');
if status ~= 0
    error('VLFeat build failed.');
end

fprintf(1, 'Running at_setup to refresh MATLAB paths and MEX builds.\n');
cd(code_root);
run('at_setup');

fprintf(1, 'Verifying resolved functions:\n');
fprintf(1, '  vl_phow: %s\n', which('vl_phow'));
fprintf(1, '  yael_kmeans: %s\n', which('yael_kmeans'));
fprintf(1, '  at_synthreproj2: %s\n', which('at_synthreproj2'));
fprintf(1, '  mh_iminterp: %s\n', which('mh_iminterp'));
end
