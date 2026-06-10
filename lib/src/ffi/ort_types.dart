// ignore_for_file: lines_longer_than_80_chars

import 'dart:ffi';

// ─────────────────────────────────────────────────────────────────────────────
// Opaque C struct handles
// ─────────────────────────────────────────────────────────────────────────────
// ONNX Runtime uses opaque pointer types (forward-declared structs).
// We model them in Dart as `final class Foo extends Opaque {}`.

/// `OrtEnv*` — the global ONNX Runtime environment.
final class OrtEnv extends Opaque {}

/// `OrtSession*` — an inference session.
final class OrtSession extends Opaque {}

/// `OrtSessionOptions*` — configuration for creating a session.
final class OrtSessionOptions extends Opaque {}

/// `OrtRunOptions*` — per-run configuration.
final class OrtRunOptions extends Opaque {}

/// `OrtValue*` — a tensor value.
final class OrtValue extends Opaque {}

/// `OrtMemoryInfo*` — describes where tensor data resides.
final class OrtMemoryInfo extends Opaque {}

/// `OrtAllocator*` — native memory allocator.
final class OrtAllocator extends Opaque {}

/// `OrtStatus*` — nullable error status (null == OK).
final class OrtStatus extends Opaque {}

/// `OrtModelMetadata*` — model metadata.
final class OrtModelMetadata extends Opaque {}

// ─────────────────────────────────────────────────────────────────────────────
// Training-specific opaque handles (onnxruntime_training_c_api.h)
// ─────────────────────────────────────────────────────────────────────────────

/// `OrtTrainingSession*` — wraps a training session.
final class OrtTrainingSession extends Opaque {}

/// `OrtCheckpointState*` — holds trainable parameter values.
final class OrtCheckpointState extends Opaque {}

// ─────────────────────────────────────────────────────────────────────────────
// C function signature typedefs
// ─────────────────────────────────────────────────────────────────────────────
//
// Convention:
//   `_XxxNative`  — the native C function signature (uses C types)
//   `XxxDart`     — the Dart callable version (uses Dart types)

// ── OrtCreateEnv ──────────────────────────────────────────────────────────
typedef OrtCreateEnvNative = Pointer<OrtStatus> Function(
  Int32 logSeverityLevel,
  Pointer<Void> logId,          // const char*
  Pointer<Pointer<OrtEnv>> out,
);
typedef OrtCreateEnvDart = Pointer<OrtStatus> Function(
  int logSeverityLevel,
  Pointer<Void> logId,
  Pointer<Pointer<OrtEnv>> out,
);

// ── OrtCreateSessionOptions ───────────────────────────────────────────────
typedef OrtCreateSessionOptionsNative = Pointer<OrtStatus> Function(
  Pointer<Pointer<OrtSessionOptions>> out,
);
typedef OrtCreateSessionOptionsDart = Pointer<OrtStatus> Function(
  Pointer<Pointer<OrtSessionOptions>> out,
);

// ── OrtCreateMemoryInfo ───────────────────────────────────────────────────
typedef OrtCreateMemoryInfoNative = Pointer<OrtStatus> Function(
  Pointer<Void> name,           // const char*
  Int32 allocatorType,
  Int32 deviceId,
  Int32 memType,
  Pointer<Pointer<OrtMemoryInfo>> out,
);
typedef OrtCreateMemoryInfoDart = Pointer<OrtStatus> Function(
  Pointer<Void> name,
  int allocatorType,
  int deviceId,
  int memType,
  Pointer<Pointer<OrtMemoryInfo>> out,
);

// ── OrtCreateTensorWithDataAsOrtValue ─────────────────────────────────────
typedef OrtCreateTensorNative = Pointer<OrtStatus> Function(
  Pointer<OrtMemoryInfo> info,
  Pointer<Void> pData,
  Uint64 dataLen,
  Pointer<Int64> shape,
  Uint64 shapeLen,
  Int32 tensorElementType,      // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1
  Pointer<Pointer<OrtValue>> out,
);
typedef OrtCreateTensorDart = Pointer<OrtStatus> Function(
  Pointer<OrtMemoryInfo> info,
  Pointer<Void> pData,
  int dataLen,
  Pointer<Int64> shape,
  int shapeLen,
  int tensorElementType,
  Pointer<Pointer<OrtValue>> out,
);

// ── OrtRun ────────────────────────────────────────────────────────────────
typedef OrtRunNative = Pointer<OrtStatus> Function(
  Pointer<OrtSession> session,
  Pointer<OrtRunOptions> runOptions,    // nullable
  Pointer<Pointer<Void>> inputNames,    // const char* const*
  Pointer<Pointer<OrtValue>> inputs,
  Uint64 inputLen,
  Pointer<Pointer<Void>> outputNames,
  Uint64 outputNamesLen,
  Pointer<Pointer<OrtValue>> outputs,
);
typedef OrtRunDart = Pointer<OrtStatus> Function(
  Pointer<OrtSession> session,
  Pointer<OrtRunOptions> runOptions,
  Pointer<Pointer<Void>> inputNames,
  Pointer<Pointer<OrtValue>> inputs,
  int inputLen,
  Pointer<Pointer<Void>> outputNames,
  int outputNamesLen,
  Pointer<Pointer<OrtValue>> outputs,
);

// ── OrtGetTensorMutableData ───────────────────────────────────────────────
typedef OrtGetTensorMutableDataNative = Pointer<OrtStatus> Function(
  Pointer<OrtValue> value,
  Pointer<Pointer<Void>> out,
);
typedef OrtGetTensorMutableDataDart = Pointer<OrtStatus> Function(
  Pointer<OrtValue> value,
  Pointer<Pointer<Void>> out,
);

// ── ORT Training API ──────────────────────────────────────────────────────
typedef OrtLoadCheckpointNative = Pointer<OrtStatus> Function(
  Pointer<Void> checkpointPath,
  Pointer<Pointer<OrtCheckpointState>> out,
);
typedef OrtLoadCheckpointDart = Pointer<OrtStatus> Function(
  Pointer<Void> checkpointPath,
  Pointer<Pointer<OrtCheckpointState>> out,
);

typedef OrtCreateTrainingSessionNative = Pointer<OrtStatus> Function(
  Pointer<OrtEnv> env,
  Pointer<OrtSessionOptions> options,
  Pointer<OrtCheckpointState> checkpoint,
  Pointer<Void> trainModelPath,
  Pointer<Void> evalModelPath,
  Pointer<Void> optimizerModelPath,
  Pointer<Pointer<OrtTrainingSession>> out,
);
typedef OrtCreateTrainingSessionDart = Pointer<OrtStatus> Function(
  Pointer<OrtEnv> env,
  Pointer<OrtSessionOptions> options,
  Pointer<OrtCheckpointState> checkpoint,
  Pointer<Void> trainModelPath,
  Pointer<Void> evalModelPath,
  Pointer<Void> optimizerModelPath,
  Pointer<Pointer<OrtTrainingSession>> out,
);

typedef OrtTrainStepNative = Pointer<OrtStatus> Function(
  Pointer<OrtTrainingSession> session,
  Pointer<OrtRunOptions> runOptions,
  Uint64 inputsLen,
  Pointer<Pointer<OrtValue>> inputs,
  Uint64 outputsLen,
  Pointer<Pointer<OrtValue>> outputs,
);
typedef OrtTrainStepDart = Pointer<OrtStatus> Function(
  Pointer<OrtTrainingSession> session,
  Pointer<OrtRunOptions> runOptions,
  int inputsLen,
  Pointer<Pointer<OrtValue>> inputs,
  int outputsLen,
  Pointer<Pointer<OrtValue>> outputs,
);

typedef OrtEvalStepNative = Pointer<OrtStatus> Function(
  Pointer<OrtTrainingSession> session,
  Pointer<OrtRunOptions> runOptions,
  Uint64 inputsLen,
  Pointer<Pointer<OrtValue>> inputs,
  Uint64 outputsLen,
  Pointer<Pointer<OrtValue>> outputs,
);
typedef OrtEvalStepDart = Pointer<OrtStatus> Function(
  Pointer<OrtTrainingSession> session,
  Pointer<OrtRunOptions> runOptions,
  int inputsLen,
  Pointer<Pointer<OrtValue>> inputs,
  int outputsLen,
  Pointer<Pointer<OrtValue>> outputs,
);

typedef OrtOptimizerStepNative = Pointer<OrtStatus> Function(
  Pointer<OrtTrainingSession> session,
  Pointer<OrtRunOptions> runOptions,
);
typedef OrtOptimizerStepDart = Pointer<OrtStatus> Function(
  Pointer<OrtTrainingSession> session,
  Pointer<OrtRunOptions> runOptions,
);

typedef OrtExportModelForInferencingNative = Pointer<OrtStatus> Function(
  Pointer<OrtTrainingSession> session,
  Pointer<Void> inferenceModelPath,
  Uint64 graphOutputNamesLen,
  Pointer<Pointer<Void>> graphOutputNames,
);
typedef OrtExportModelForInferencingDart = Pointer<OrtStatus> Function(
  Pointer<OrtTrainingSession> session,
  Pointer<Void> inferenceModelPath,
  int graphOutputNamesLen,
  Pointer<Pointer<Void>> graphOutputNames,
);

// ── Release helpers ───────────────────────────────────────────────────────
typedef OrtReleaseEnvNative = Void Function(Pointer<OrtEnv>);
typedef OrtReleaseEnvDart = void Function(Pointer<OrtEnv>);

typedef OrtReleaseSessionNative = Void Function(Pointer<OrtSession>);
typedef OrtReleaseSessionDart = void Function(Pointer<OrtSession>);

typedef OrtReleaseValueNative = Void Function(Pointer<OrtValue>);
typedef OrtReleaseValueDart = void Function(Pointer<OrtValue>);

typedef OrtReleaseMemoryInfoNative = Void Function(Pointer<OrtMemoryInfo>);
typedef OrtReleaseMemoryInfoDart = void Function(Pointer<OrtMemoryInfo>);

typedef OrtReleaseStatusNative = Void Function(Pointer<OrtStatus>);
typedef OrtReleaseStatusDart = void Function(Pointer<OrtStatus>);

typedef OrtReleaseTrainingSessionNative = Void Function(
  Pointer<OrtTrainingSession>,
);
typedef OrtReleaseTrainingSessionDart = void Function(
  Pointer<OrtTrainingSession>,
);

typedef OrtReleaseCheckpointStateNative = Void Function(
  Pointer<OrtCheckpointState>,
);
typedef OrtReleaseCheckpointStateDart = void Function(
  Pointer<OrtCheckpointState>,
);

// ── Error message ─────────────────────────────────────────────────────────
typedef OrtGetErrorMessageNative = Pointer<Void> Function(
  Pointer<OrtStatus> status,
);
typedef OrtGetErrorMessageDart = Pointer<Void> Function(
  Pointer<OrtStatus> status,
);

typedef OrtGetErrorCodeNative = Int32 Function(Pointer<OrtStatus> status);
typedef OrtGetErrorCodeDart = int Function(Pointer<OrtStatus> status);

// ─────────────────────────────────────────────────────────────────────────────
// ONNX tensor element type constants
// ─────────────────────────────────────────────────────────────────────────────
abstract class OrtTensorElementDataType {
  static const int undefined = 0;
  static const int float = 1;  // float32
  static const int uint8 = 2;
  static const int int8 = 3;
  static const int uint16 = 4;
  static const int int16 = 5;
  static const int int32 = 6;
  static const int int64 = 7;
  static const int string = 8;
  static const int bool_ = 9;
  static const int float16 = 10;
  static const int double_ = 11;
}

// OrtAllocatorType
abstract class OrtAllocatorType {
  static const int invalid = -1;
  static const int device = 0;
  static const int arena = 1;
}

// OrtMemType
abstract class OrtMemType {
  static const int cpuInput = -2;
  static const int cpuOutput = -1;
  static const int cpu = 0;
  static const int defaultMemory = 0;
}

// OrtLoggingLevel
abstract class OrtLoggingLevel {
  static const int verbose = 0;
  static const int info = 1;
  static const int warning = 2;
  static const int error = 3;
  static const int fatal = 4;
}
