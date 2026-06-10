#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_ort_libs.sh
#
# Downloads pre-built ONNX Runtime (with On-Device Training) native libraries
# for Android and iOS, and places them in the expected paths.
#
# Usage:
#   chmod +x tool/download_ort_libs.sh
#   ./tool/download_ort_libs.sh
#
# Requirements: curl, unzip (usually pre-installed on macOS/Linux)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
ORT_VERSION="1.18.0"
NATIVE_DIR="native/onnxruntime"

ANDROID_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-android-training-${ORT_VERSION}.zip"
IOS_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-training-ios-${ORT_VERSION}.zip"

# ── Helper ─────────────────────────────────────────────────────────────────
info()  { echo -e "\033[0;36m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[0;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[0;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[0;31m[ERR ]\033[0m  $*"; exit 1; }

check_cmd() {
    command -v "$1" >/dev/null 2>&1 || err "'$1' not found. Please install it."
}

# ── Pre-flight ──────────────────────────────────────────────────────────────
check_cmd curl
check_cmd unzip

mkdir -p "${NATIVE_DIR}/lib/arm64-v8a"
mkdir -p "${NATIVE_DIR}/lib/armeabi-v7a"
mkdir -p "${NATIVE_DIR}/lib/x86_64"
mkdir -p "${NATIVE_DIR}/include"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

# ── Android ─────────────────────────────────────────────────────────────────
info "Downloading ORT ${ORT_VERSION} for Android…"
ANDROID_ZIP="${TMP_DIR}/ort-android.zip"

if curl -fsSL --progress-bar -o "${ANDROID_ZIP}" "${ANDROID_URL}"; then
    ok "Downloaded Android ORT package"
else
    warn "Failed to download Android ORT from GitHub."
    warn "Please download manually from:"
    warn "  ${ANDROID_URL}"
    warn "And extract libonnxruntime.so to native/onnxruntime/lib/<ABI>/"
    warn "And onnxruntime_c_api.h + onnxruntime_training_c_api.h to native/onnxruntime/include/"
fi

if [[ -f "${ANDROID_ZIP}" ]]; then
    info "Extracting Android libraries…"
    ANDROID_EXTRACT="${TMP_DIR}/android"
    mkdir -p "${ANDROID_EXTRACT}"
    unzip -q "${ANDROID_ZIP}" -d "${ANDROID_EXTRACT}"

    # Copy headers
    find "${ANDROID_EXTRACT}" -name "onnxruntime_c_api.h" -exec cp {} "${NATIVE_DIR}/include/" \;
    find "${ANDROID_EXTRACT}" -name "onnxruntime_training_c_api.h" -exec cp {} "${NATIVE_DIR}/include/" \;

    # Copy .so files per ABI
    for ABI in arm64-v8a armeabi-v7a x86_64; do
        SO_SRC=$(find "${ANDROID_EXTRACT}" -path "*${ABI}*" -name "libonnxruntime*.so" | head -1)
        if [[ -n "${SO_SRC}" ]]; then
            cp "${SO_SRC}" "${NATIVE_DIR}/lib/${ABI}/libonnxruntime.so"
            ok "Copied libonnxruntime.so → ${ABI}"
        else
            warn "libonnxruntime.so not found for ${ABI}"
        fi
    done
fi

# ── iOS ─────────────────────────────────────────────────────────────────────
IOS_FRAMEWORK_DIR="ios/Frameworks"
mkdir -p "${IOS_FRAMEWORK_DIR}"

info "Downloading ORT ${ORT_VERSION} for iOS…"
IOS_ZIP="${TMP_DIR}/ort-ios.zip"

if curl -fsSL --progress-bar -o "${IOS_ZIP}" "${IOS_URL}"; then
    ok "Downloaded iOS ORT package"
    info "Extracting iOS xcframework…"
    unzip -q "${IOS_ZIP}" -d "${TMP_DIR}/ios"
    XC=$(find "${TMP_DIR}/ios" -name "*.xcframework" | head -1)
    if [[ -n "${XC}" ]]; then
        cp -r "${XC}" "${IOS_FRAMEWORK_DIR}/onnxruntime.xcframework"
        ok "Copied onnxruntime.xcframework → ios/Frameworks/"
    else
        warn "xcframework not found in iOS zip"
    fi
else
    warn "Failed to download iOS ORT from GitHub."
    warn "Please download manually from:"
    warn "  ${IOS_URL}"
    warn "And extract onnxruntime.xcframework to ios/Frameworks/"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────"
echo " Download complete. Expected file tree:"
echo ""
echo " native/onnxruntime/"
echo "   include/"
echo "     onnxruntime_c_api.h"
echo "     onnxruntime_training_c_api.h"
echo "   lib/"
echo "     arm64-v8a/libonnxruntime.so"
echo "     armeabi-v7a/libonnxruntime.so"
echo "     x86_64/libonnxruntime.so"
echo ""
echo " ios/Frameworks/"
echo "   onnxruntime.xcframework/"
echo "────────────────────────────────────────────────"
echo ""
info "Next steps:"
echo "  1. Run: python tool/generate_ort_artifacts.py --model_type all"
echo "  2. Run: flutter build apk (or xcodebuild for iOS)"
