start_path = pwd;

fprintf(1, '   Adding paths for dense-vlad demo,\n   root dir %s.\n', start_path);
addpath(start_path);

%--- vlfeat
if ~exist('vl_version','file'), 
  run([start_path '/thirdparty/vlfeat-0.9.20/toolbox/vl_setup']);
end;

%--- yael library
switch computer
  case 'GLNXA64'
    addpath([start_path '/thirdparty/yael_matlab_linux64_v438']);
  case 'MACI64'
    addpath([start_path '/thirdparty/yael_matlab_mac64_v438']);
  otherwise
    error('Yael library is missing!')
end

%--- our codes
addpath([start_path '/code']);
if ~exist(['at_synthreproj2.' mexext],'file')
  cd([start_path '/code']);
  mex at_synthreproj2.c;
  cd(start_path);
end
if ~exist(['mh_iminterp.' mexext],'file')
  cd([start_path '/code']);
  mex mh_iminterp.c
  cd(start_path);
end

