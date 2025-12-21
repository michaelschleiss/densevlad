#!/usr/bin/env bash
# Download and extract the original 247code.zip from the authors
# This contains the MATLAB implementation and reference data files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$PROJECT_ROOT"
ZIP_FILE="$TARGET_DIR/247code.zip"
URL="http://www.ok.ctrl.titech.ac.jp/~torii/project/247/download/247code.zip"

echo "Downloading original 247code package..."
echo "URL: $URL"
echo "Target: $ZIP_FILE"
echo ""

# Download if not exists
if [ -f "$ZIP_FILE" ]; then
    echo "Archive already exists: $ZIP_FILE"
    echo "Delete it first if you want to re-download."
else
    echo "Downloading..."
    if command -v curl &> /dev/null; then
        curl -L -o "$ZIP_FILE" "$URL"
    elif command -v wget &> /dev/null; then
        wget -O "$ZIP_FILE" "$URL"
    else
        echo "Error: Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    echo "Download complete: $ZIP_FILE"
fi

# Extract
echo ""
echo "Extracting..."
unzip -q "$ZIP_FILE" -d "$TARGET_DIR"
echo "Extracted to: $TARGET_DIR/247code"
rm -f "$ZIP_FILE"

echo ""
echo "Contents:"
ls -lh "$TARGET_DIR/247code/" | head -20

echo ""
echo "Done! The original 247code is available at:"
echo "  $TARGET_DIR/247code/"
echo ""
echo "Key directories:"
echo "  - code/         : Original MATLAB implementation"
echo "  - data/         : Vocabulary, PCA matrices, example images"
echo "  - thirdparty/   : VLFeat 0.9.20 (original version)"
