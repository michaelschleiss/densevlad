from __future__ import annotations

# Reference-only helper; applies an optional cyvlfeat dsift patch for speed.

from pathlib import Path
import sys


def _patch_cysift(path: Path) -> bool:
    text = path.read_text()
    if "contrast_threshold" in text and "permute_arr" in text:
        return False

    if "from libc.string cimport memset" not in text:
        text = text.replace(
            "from libc.stdlib cimport qsort\n",
            "from libc.stdlib cimport qsort\nfrom libc.string cimport memset\n",
            1,
        )

    sig_old = (
        "cpdef cy_dsift(float[:, ::1] data, int[:] step,\n"
        "               int[:] size, int[:] bounds, float window_size, bint norm,\n"
        "               bint fast, bint float_descriptors, int[:] geometry,\n"
        "               bint verbose):\n"
    )
    sig_new = (
        "cpdef cy_dsift(float[:, ::1] data, int[:] step,\n"
        "               int[:] size, int[:] bounds, float window_size, bint norm,\n"
        "               bint fast, bint float_descriptors, int[:] geometry,\n"
        "               bint verbose, float contrast_threshold=-1.0,\n"
        "               object permute=None):\n"
    )
    if sig_old in text:
        text = text.replace(sig_old, sig_new, 1)

    needle = (
        "float[:, ::1] out_descriptors\n"
        "        float[:, ::1] out_frames\n"
    )
    if needle in text:
        text = text.replace(
            needle,
            "np.ndarray out_descriptors\n"
            "        float[:, ::1] out_frames\n"
            "        float *out_f_ptr\n"
            "        unsigned char *out_u8_ptr\n",
            1,
        )

    needle = (
        "int ndims = 0\n"
        "        int descriptor_index = 0\n"
        "        float* linear_descriptor\n"
    )
    if needle in text:
        text = text.replace(
            needle,
            "int ndims = 0\n"
            "        int descriptor_length = 0\n"
            "        int total = 0\n"
            "        int idx = 0\n"
            "        int src_i = 0\n"
            "        int perm_len = 0\n"
            "        int row_offset = 0\n"
            "        float val\n",
            1,
        )

    if "permute_arr" not in text:
        text = text.replace(
            "        unsigned char *out_u8_ptr\n",
            "        unsigned char *out_u8_ptr\n"
            "        np.ndarray permute_arr\n"
            "        int *perm_ptr\n",
            1,
        )

    text = text.replace(
        "    vl_dsift_process(dsift, &data[0, 0])\n",
        "    with nogil:\n        vl_dsift_process(dsift, &data[0, 0])\n",
        1,
    )

    output_block = (
        "    # Create output arrays\n"
        "    out_descriptors = np.empty((num_frames, descriptor_length),\n"
        "                               dtype=np.float32, order='C')\n"
        "    # Grab the pointer to the data so we can walk it linearly\n"
        "    linear_descriptor = &out_descriptors[0, 0]\n"
        "\n"
        "    # The norm is added as the third component if set\n"
        "    if norm:\n"
        "        ndims = 3\n"
        "        out_frames = np.empty((num_frames, ndims), dtype=np.float32)\n"
        "    else:\n"
        "        ndims = 2\n"
        "        out_frames = np.empty((num_frames, ndims), dtype=np.float32)\n"
        "\n"
        "    # Copy results out\n"
        "    for k in range(num_frames):\n"
        "        out_frames[k, 0] = frames_array[k].y\n"
        "        out_frames[k, 1] = frames_array[k].x\n"
        "\n"
        "        # We have an implied / 2 in the norm, because of the clipping below\n"
        "        if norm:\n"
        "            out_frames[k, 2] = frames_array[k].norm\n"
        "\n"
        "        # We don't need to transpose because our memory is in the correct\n"
        "        # order already!\n"
        "        for i in range(descriptor_length):\n"
        "            descriptor_index = num_frames * i + k\n"
        "            linear_descriptor[descriptor_index] = \\\n"
        "                min(512.0 * descriptors_array[descriptor_index], 255.0)\n"
        "\n"
        "    # Clean up the allocated memory\n"
        "    vl_dsift_delete(dsift)\n"
        "\n"
        "    if float_descriptors:\n"
        "        return np.asarray(out_frames), np.asarray(out_descriptors)\n"
        "    else:\n"
        "        return np.asarray(out_frames), np.asarray(out_descriptors).astype(np.uint8)\n"
    )

    output_replacement = (
        "    # Create output arrays\n"
        "    if float_descriptors:\n"
        "        out_descriptors = np.empty((num_frames, descriptor_length),\n"
        "                                   dtype=np.float32, order='C')\n"
        "        out_f_ptr = <float *> out_descriptors.data\n"
        "    else:\n"
        "        out_descriptors = np.empty((num_frames, descriptor_length),\n"
        "                                   dtype=np.uint8, order='C')\n"
        "        out_u8_ptr = <unsigned char *> out_descriptors.data\n"
        "\n"
        "    # The norm is added as the third component if set\n"
        "    if norm:\n"
        "        ndims = 3\n"
        "        out_frames = np.empty((num_frames, ndims), dtype=np.float32)\n"
        "    else:\n"
        "        ndims = 2\n"
        "        out_frames = np.empty((num_frames, ndims), dtype=np.float32)\n"
        "\n"
        "    # Copy frames out\n"
        "    for k in range(num_frames):\n"
        "        out_frames[k, 0] = frames_array[k].y\n"
        "        out_frames[k, 1] = frames_array[k].x\n"
        "        if norm:\n"
        "            out_frames[k, 2] = frames_array[k].norm\n"
        "\n"
        "    if contrast_threshold >= 0 and not norm:\n"
        "        raise ValueError(\"contrast_threshold requires norm=True\")\n"
        "\n"
        "    permute_arr = None\n"
        "    perm_ptr = <int *> 0\n"
        "    perm_len = 0\n"
        "    if permute is not None:\n"
        "        permute_arr = np.ascontiguousarray(permute, dtype=np.int32)\n"
        "        if permute_arr.ndim != 1:\n"
        "            raise ValueError(\"permute must be 1D\")\n"
        "        perm_len = permute_arr.shape[0]\n"
        "        if perm_len != descriptor_length:\n"
        "            raise ValueError(\"permute length must match descriptor size\")\n"
        "        perm_ptr = <int *> permute_arr.data\n"
        "\n"
        "    total = num_frames * descriptor_length\n"
        "    if perm_ptr == <int *> 0 and contrast_threshold < 0:\n"
        "        if float_descriptors:\n"
        "            with nogil:\n"
        "                for idx in range(total):\n"
        "                    val = descriptors_array[idx] * 512.0\n"
        "                    if val > 255.0:\n"
        "                        val = 255.0\n"
        "                    out_f_ptr[idx] = val\n"
        "        else:\n"
        "            with nogil:\n"
        "                for idx in range(total):\n"
        "                    val = descriptors_array[idx] * 512.0\n"
        "                    if val > 255.0:\n"
        "                        val = 255.0\n"
        "                    out_u8_ptr[idx] = <unsigned char> val\n"
        "    else:\n"
        "        if float_descriptors:\n"
        "            with nogil:\n"
        "                for k in range(num_frames):\n"
        "                    row_offset = k * descriptor_length\n"
        "                    if contrast_threshold >= 0 and frames_array[k].norm < contrast_threshold:\n"
        "                        memset(out_f_ptr + row_offset, 0, descriptor_length * sizeof(float))\n"
        "                        continue\n"
        "                    if perm_ptr == <int *> 0:\n"
        "                        for idx in range(descriptor_length):\n"
        "                            val = descriptors_array[row_offset + idx] * 512.0\n"
        "                            if val > 255.0:\n"
        "                                val = 255.0\n"
        "                            out_f_ptr[row_offset + idx] = val\n"
        "                    else:\n"
        "                        for idx in range(descriptor_length):\n"
        "                            src_i = perm_ptr[idx]\n"
        "                            val = descriptors_array[row_offset + src_i] * 512.0\n"
        "                            if val > 255.0:\n"
        "                                val = 255.0\n"
        "                            out_f_ptr[row_offset + idx] = val\n"
        "        else:\n"
        "            with nogil:\n"
        "                for k in range(num_frames):\n"
        "                    row_offset = k * descriptor_length\n"
        "                    if contrast_threshold >= 0 and frames_array[k].norm < contrast_threshold:\n"
        "                        memset(out_u8_ptr + row_offset, 0, descriptor_length * sizeof(unsigned char))\n"
        "                        continue\n"
        "                    if perm_ptr == <int *> 0:\n"
        "                        for idx in range(descriptor_length):\n"
        "                            val = descriptors_array[row_offset + idx] * 512.0\n"
        "                            if val > 255.0:\n"
        "                                val = 255.0\n"
        "                            out_u8_ptr[row_offset + idx] = <unsigned char> val\n"
        "                    else:\n"
        "                        for idx in range(descriptor_length):\n"
        "                            src_i = perm_ptr[idx]\n"
        "                            val = descriptors_array[row_offset + src_i] * 512.0\n"
        "                            if val > 255.0:\n"
        "                                val = 255.0\n"
        "                            out_u8_ptr[row_offset + idx] = <unsigned char> val\n"
        "\n"
        "    # Clean up the allocated memory\n"
        "    vl_dsift_delete(dsift)\n"
        "\n"
        "    return np.asarray(out_frames), np.asarray(out_descriptors)\n"
    )

    if output_block in text:
        text = text.replace(output_block, output_replacement, 1)

    path.write_text(text)
    return True


def _patch_dsift_pxd(path: Path) -> bool:
    text = path.read_text()
    needle = "    void vl_dsift_process(VlDsiftFilter *self, float*im)\n"
    if "vl_dsift_process(VlDsiftFilter *self, float*im) nogil" in text:
        return False
    if needle not in text:
        raise SystemExit("Expected vl_dsift_process declaration not found in dsift.pxd")
    text = text.replace(needle, "    void vl_dsift_process(VlDsiftFilter *self, float*im) nogil\n", 1)
    path.write_text(text)
    return True


def _patch_dsift_py(path: Path) -> bool:
    text = path.read_text()
    if "contrast_threshold" in text and "permute" in text:
        return False

    sig_old = (
        "def dsift(image, step=1, size=3, bounds=None, window_size=-1, norm=False,\n"
        "          fast=False, float_descriptors=False, geometry=(4, 4, 8),\n"
        "          verbose=False):\n"
    )
    sig_new = (
        "def dsift(image, step=1, size=3, bounds=None, window_size=-1, norm=False,\n"
        "          fast=False, float_descriptors=False, geometry=(4, 4, 8),\n"
        "          verbose=False, contrast_threshold=None, permute=None):\n"
    )
    if sig_old in text:
        text = text.replace(sig_old, sig_new, 1)

    call_old = (
        "    frames, descriptors = cy_dsift(image, step, size, bounds, window_size,\n"
        "                                   norm, fast, float_descriptors, geometry,\n"
        "                                   verbose)\n"
    )
    call_new = (
        "    if contrast_threshold is None:\n"
        "        contrast_threshold = -1.0\n"
        "    else:\n"
        "        contrast_threshold = float(contrast_threshold)\n"
        "    if permute is not None:\n"
        "        permute = np.asarray(permute, dtype=np.int32)\n"
        "    frames, descriptors = cy_dsift(image, step, size, bounds, window_size,\n"
        "                                   norm, fast, float_descriptors, geometry,\n"
        "                                   verbose, contrast_threshold, permute)\n"
    )
    if call_old in text:
        text = text.replace(call_old, call_new, 1)

    path.write_text(text)
    return True


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_cyvlfeat_sdist.py /path/to/cyvlfeat-src")

    src_dir = Path(sys.argv[1])
    cysift = src_dir / "cyvlfeat" / "sift" / "cysift.pyx"
    if not cysift.exists():
        raise SystemExit(f"Missing cysift.pyx at {cysift}")

    dsift_pxd = src_dir / "cyvlfeat" / "_vl" / "dsift.pxd"
    if not dsift_pxd.exists():
        raise SystemExit(f"Missing dsift.pxd at {dsift_pxd}")
    dsift_py = src_dir / "cyvlfeat" / "sift" / "dsift.py"
    if not dsift_py.exists():
        raise SystemExit(f"Missing dsift.py at {dsift_py}")

    patched_cysift = _patch_cysift(cysift)
    patched_pxd = _patch_dsift_pxd(dsift_pxd)
    patched_py = _patch_dsift_py(dsift_py)

    if patched_cysift or patched_pxd or patched_py:
        print("Patched cyvlfeat dsift sources for DVlad optimizations.")
    else:
        print("cyvlfeat dsift sources already patched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
