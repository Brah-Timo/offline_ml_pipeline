// ignore_for_file: lines_longer_than_80_chars

import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'ort_types.dart';
import 'native_memory.dart';
import '../utils/error_handler.dart';

// ─────────────────────────────────────────────────────────────────────────────
// OrtBindings — Dart ↔ ONNX Runtime C API bridge
// ─────────────────────────────────────────────────────────────────────────────

/// Singleton that loads the ONNX Runtime native shared library and exposes
/// its C API as Dart callable functions.
///
/// Uses `dart:ffi` to look up every function symbol in the loaded
/// [DynamicLibrary] and wraps them with the correct Dart call signatures.
///
/// Acquiring the singleton:
/// ```dart
/// final ort = OrtBindings.instance;
/// ```
///
/// The library is loaded lazily on first access.
class OrtBindings {
  // ── Singleton ──────────────────────────────────────────────────────────
  static OrtBindings? _instance;

  static OrtBindings get instance {
    _instance ??= OrtBindings._load();
    return _instance!;
  }

  // Private constructor — loads the shared library
  OrtBindings._load() {
    _lib = _openLibrary();
    _bind();
  }

  late final DynamicLibrary _lib;

  // ── Bound C functions ──────────────────────────────────────────────────

  // Environment
  late final OrtCreateEnvDart createEnv;

  // Session options
  late final OrtCreateSessionOptionsDart createSessionOptions;

  // Memory info
  late final OrtCreateMemoryInfoDart createMemoryInfo;

  // Tensors
  late final OrtCreateTensorDart createTensorWithData;
  late final OrtGetTensorMutableDataDart getTensorMutableData;

  // Run (inference)
  late final OrtRunDart run;

  // Training
  late final OrtLoadCheckpointDart loadCheckpoint;
  late final OrtCreateTrainingSessionDart createTrainingSession;
  late final OrtTrainStepDart trainStep;
  late final OrtEvalStepDart evalStep;
  late final OrtOptimizerStepDart optimizerStep;
  late final OrtExportModelForInferencingDart exportModelForInferencing;

  // Error handling
  late final OrtGetErrorMessageDart getErrorMessage;
  late final OrtGetErrorCodeDart getErrorCode;

  // Release / cleanup
  late final OrtReleaseEnvDart releaseEnv;
  late final OrtReleaseSessionDart releaseSession;
  late final OrtReleaseValueDart releaseValue;
  late final OrtReleaseMemoryInfoDart releaseMemoryInfo;
  late final OrtReleaseStatusDart releaseStatus;
  late final OrtReleaseTrainingSessionDart releaseTrainingSession;
  late final OrtReleaseCheckpointStateDart releaseCheckpointState;

  // ── Helper: check ORT status ───────────────────────────────────────────

  /// Checks [status]; throws [OrtException] if it indicates an error.
  void check(Pointer<OrtStatus> status) {
    if (status == nullptr) return;

    final msg = NativeMemory.stringFromVoid(getErrorMessage(status));
    final code = getErrorCode(status);
    releaseStatus(status);

    throw OrtException(msg, errorCode: code);
  }

  // ── Library loading ────────────────────────────────────────────────────

  static DynamicLibrary _openLibrary() {
    if (Platform.isAndroid) {
      return DynamicLibrary.open('libonnxruntime.so');
    } else if (Platform.isIOS) {
      // On iOS the framework is statically linked into the process.
      return DynamicLibrary.process();
    } else if (Platform.isMacOS) {
      return DynamicLibrary.open('libonnxruntime.dylib');
    } else if (Platform.isWindows) {
      return DynamicLibrary.open('onnxruntime.dll');
    } else if (Platform.isLinux) {
      return DynamicLibrary.open('libonnxruntime.so');
    }
    throw UnsupportedPlatformException(Platform.operatingSystem);
  }

  // ── Symbol binding ─────────────────────────────────────────────────────

  void _bind() {
    // ── Environment ──────────────────────────────────────────────────────
    createEnv = _lib.lookupFunction<OrtCreateEnvNative, OrtCreateEnvDart>(
      'OrtCreateEnv',
    );

    createSessionOptions = _lib.lookupFunction<
        OrtCreateSessionOptionsNative,
        OrtCreateSessionOptionsDart>('OrtCreateSessionOptions');

    // ── Memory ────────────────────────────────────────────────────────────
    createMemoryInfo = _lib.lookupFunction<
        OrtCreateMemoryInfoNative,
        OrtCreateMemoryInfoDart>('OrtCreateMemoryInfo');

    // ── Tensors ───────────────────────────────────────────────────────────
    createTensorWithData = _lib.lookupFunction<
        OrtCreateTensorNative,
        OrtCreateTensorDart>('OrtCreateTensorWithDataAsOrtValue');

    getTensorMutableData = _lib.lookupFunction<
        OrtGetTensorMutableDataNative,
        OrtGetTensorMutableDataDart>('OrtGetTensorMutableData');

    // ── Inference run ─────────────────────────────────────────────────────
    run = _lib.lookupFunction<OrtRunNative, OrtRunDart>('OrtRun');

    // ── Training API ──────────────────────────────────────────────────────
    // These symbols live in onnxruntime_training_c_api.h
    loadCheckpoint = _lib.lookupFunction<
        OrtLoadCheckpointNative,
        OrtLoadCheckpointDart>('OrtTrainingLoadCheckpoint');

    createTrainingSession = _lib.lookupFunction<
        OrtCreateTrainingSessionNative,
        OrtCreateTrainingSessionDart>('OrtTrainingCreateTrainingSession');

    trainStep = _lib.lookupFunction<OrtTrainStepNative, OrtTrainStepDart>(
      'OrtTrainingTrainStep',
    );

    evalStep = _lib.lookupFunction<OrtEvalStepNative, OrtEvalStepDart>(
      'OrtTrainingEvalStep',
    );

    optimizerStep = _lib.lookupFunction<
        OrtOptimizerStepNative,
        OrtOptimizerStepDart>('OrtTrainingOptimizerStep');

    exportModelForInferencing = _lib.lookupFunction<
        OrtExportModelForInferencingNative,
        OrtExportModelForInferencingDart>(
      'OrtTrainingExportModelForInferencing',
    );

    // ── Error helpers ─────────────────────────────────────────────────────
    getErrorMessage = _lib.lookupFunction<
        OrtGetErrorMessageNative,
        OrtGetErrorMessageDart>('OrtGetErrorMessage');

    getErrorCode = _lib.lookupFunction<
        OrtGetErrorCodeNative,
        OrtGetErrorCodeDart>('OrtGetErrorCode');

    // ── Release helpers ───────────────────────────────────────────────────
    releaseEnv = _lib.lookupFunction<
        OrtReleaseEnvNative, OrtReleaseEnvDart>('OrtReleaseEnv');

    releaseSession = _lib.lookupFunction<
        OrtReleaseSessionNative,
        OrtReleaseSessionDart>('OrtReleaseSession');

    releaseValue = _lib.lookupFunction<
        OrtReleaseValueNative,
        OrtReleaseValueDart>('OrtReleaseValue');

    releaseMemoryInfo = _lib.lookupFunction<
        OrtReleaseMemoryInfoNative,
        OrtReleaseMemoryInfoDart>('OrtReleaseMemoryInfo');

    releaseStatus = _lib.lookupFunction<
        OrtReleaseStatusNative,
        OrtReleaseStatusDart>('OrtReleaseStatus');

    releaseTrainingSession = _lib.lookupFunction<
        OrtReleaseTrainingSessionNative,
        OrtReleaseTrainingSessionDart>('OrtTrainingReleaseTrainingSession');

    releaseCheckpointState = _lib.lookupFunction<
        OrtReleaseCheckpointStateNative,
        OrtReleaseCheckpointStateDart>('OrtTrainingReleaseCheckpointState');
  }

  // ── Convenience: build a float32 OrtValue tensor ──────────────────────

  /// Creates a float32 ORT tensor from [data] with the given [shape].
  ///
  /// Caller is responsible for releasing the returned [OrtValue] via
  /// [releaseValue] when done.
  Pointer<OrtValue> buildFloat32Tensor({
    required List<double> data,
    required List<int> shape,
    required Arena arena,
  }) {
    // Allocate CPU memory info
    final memInfoPtr = arena.allocate<Pointer<OrtMemoryInfo>>(1);
    final cpuName = 'Cpu'.toNativeUtf8(allocator: arena).cast<Void>();

    check(
      createMemoryInfo(
        cpuName,
        OrtAllocatorType.arena,
        0,
        OrtMemType.cpu,
        memInfoPtr,
      ),
    );

    final dataPtr = NativeMemory.float32VoidFromList(data, arena);
    final shapePtr = NativeMemory.int64FromList(shape, arena);
    final ortValuePtr = arena.allocate<Pointer<OrtValue>>(1);

    check(
      createTensorWithData(
        memInfoPtr.value,
        dataPtr,
        data.length * 4,       // dataLen in bytes (float32 = 4 bytes)
        shapePtr,
        shape.length,
        OrtTensorElementDataType.float, // FLOAT
        ortValuePtr,
      ),
    );

    return ortValuePtr.value;
  }

  /// Reads the float32 output tensor into a Dart [List<double>].
  List<double> readFloat32Tensor(
    Pointer<OrtValue> tensor,
    int length,
    Arena arena,
  ) {
    final dataPtr = arena.allocate<Pointer<Void>>(1);
    check(getTensorMutableData(tensor, dataPtr));
    return NativeMemory.float32ToList(
      dataPtr.value.cast<Float>(),
      length,
    );
  }
}
