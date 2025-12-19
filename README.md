# dvlad

Work-in-progress Python reimplementation of the DenseVLAD baseline from:
*A. Torii, R. Arandjelović, J. Sivic, M. Okutomi, T. Pajdla — “24/7 Place Recognition by View Synthesis”, CVPR 2015.*

Initial focus: exact replication of Torii15 DenseVLAD pipeline and assets.

## Notes on dependencies

`cyvlfeat` requires the VLFeat C library (headers + `libvl`) to be installed
on your system. On Apple Silicon, build VLFeat from source with SSE/AVX
disabled and set include/library paths when installing `cyvlfeat`.

Example (paths vary):
```
CFLAGS="-I/path/to/vlfeat" LDFLAGS="-L/path/to/vlfeat/bin/maci64" \
  pip install cyvlfeat
```

For Apple Silicon without Rosetta, use `scripts/build_vlfeat_arm64.sh` to
download + patch VLFeat 0.9.20, build `libvl`, and print the exact
environment variables needed to install `cyvlfeat`.

If you use pixi, install the environment while skipping `cyvlfeat`,
then install it manually after VLFeat is built:
```
pixi install --skip-with-deps cyvlfeat
pixi run -- env PIP_NO_BUILD_ISOLATION=1 \
  CFLAGS="-I/path/to/vlfeat" LDFLAGS="-L/path/to/vlfeat/bin/maci64" \
  python -m pip install cyvlfeat
```

Pixi tasks are available:
```
pixi install --skip-with-deps cyvlfeat
pixi run build-vlfeat
pixi run install-cyvlfeat
pixi run test-prepca
```
