# FFI Bindings

`offline_ml_pipeline` uses `dart:ffi` to call the ONNX Runtime On-Device Training C API (`onnxruntime_training_c_api.h`).

---

## Files

| File | Role |
|------|------|
| `lib/src/ffi/ort_types.dart` | Opaque FFI structs + function typedef pairs |
| `lib/src/ffi/ort_bindings.dart` | `OrtBindings` singleton — loads the native library and resolves all function symbols |
| `lib/src/ffi/native_memory.dart` | Arena helpers for allocating native int64 arrays |

---

## Opaque Handle Types (`ort_types.dart`)

ONNX Runtime uses forward-declared C structs (opaque pointers). In Dart FFI, these are modelled as:

```dart
final class OrtEnv extends Opaque {}
final class OrtSession extends Opaque {}
final class OrtSessionOptions extends Opaque {}
final class OrtRunOptions extends Opaque {}
final class OrtValue extends Opaque {}
final class OrtMemoryInfo extends Opaque {}
final class OrtAllocator extends Opaque {}
final class OrtStatus extends Opaque {}
final class OrtModelMetadata extends Opaque {}

// Training-specific
final class OrtTrainingSession extends Opaque {}
final class OrtCheckpointState extends Opaque {}
```

`final class Foo extends Opaque {}` is required so that `Pointer<Foo>` satisfies the `Pointer<T extends NativeType>` bound.

> **Important**: Do NOT define a Dart class with the same name as an FFI opaque type in the same file. The Dart class will shadow the FFI type and break `Pointer<T>` resolution. The `_OrtBackendSession` training class is named with an underscore prefix precisely to avoid conflicting with `OrtTrainingSession`.

---

## Function Typedefs

Each C function has two typedefs: the native (`_Native`) and Dart-callable (`Dart`) variants.

Example — `OrtCreateEnv`:

```dart
// C signature:
// OrtStatus* OrtCreateEnv(
//   int logSeverityLevel,
//   const char* logId,
//   OrtEnv** out
// );

typedef OrtCreateEnvNative = Pointer<OrtStatus> Function(
  Int32 logSeverityLevel,
  Pointer<Void> logId,
  Pointer<Pointer<OrtEnv>> out,
);
typedef OrtCreateEnvDart = Pointer<OrtStatus> Function(
  int logSeverityLevel,
  Pointer<Void> logId,
  Pointer<Pointer<OrtEnv>> out,
);
```

The convention is consistent across all 30+ API functions in the file.

---

## OrtBindings Singleton

`OrtBindings` loads the native library once and exposes all resolved function pointers as Dart-callable fields:

```dart
class OrtBindings {
  static late OrtBindings _instance;
  static OrtBindings get instance => _instance;

  // Loaded via DynamicLibrary.open(...)
  late final OrtCreateEnvDart createEnv;
  late final OrtCreateSessionOptionsDart createSessionOptions;
  late final OrtLoadCheckpointDart loadCheckpoint;
  late final OrtCreateTrainingSessionDart createTrainingSession;
  late final OrtTrainStepDart trainStep;
  late final OrtOptimizerStepDart optimizerStep;
  late final OrtEvalStepDart evalStep;
  late final OrtExportModelForInferencingDart exportModelForInferencing;
  late final OrtReleaseTrainingSessionDart releaseTrainingSession;
  late final OrtReleaseCheckpointStateDart releaseCheckpointState;
  late final OrtReleaseEnvDart releaseEnv;
  late final OrtReleaseValueDart releaseValue;
  // ... (all other API functions)

  // Status check — throws if status != null (== OK)
  void check(Pointer<OrtStatus> status) { ... }

  // Helper: create a float32 OrtValue tensor from a Dart list
  Pointer<OrtValue> buildFloat32Tensor({
    required List<double> data,
    required List<int> shape,
    required Arena arena,
  }) { ... }

  // Helper: read float32 OrtValue tensor into a Dart list
  List<double> readFloat32Tensor(
    Pointer<OrtValue> tensor, int count, Arena arena,
  ) { ... }
}
```

### Loading the native library

```dart
// Android / Linux
DynamicLibrary.open('libonnxruntime_training.so')

// iOS / macOS
DynamicLibrary.open('libonnxruntime_training.dylib')

// Windows
DynamicLibrary.open('onnxruntime_training.dll')
```

`OrtBindings.instance` throws if the library cannot be opened — this is caught by `TrainingSession.create()` to fall back to pure-Dart mode.

---

## NativeMemory Helpers

`NativeMemory` provides allocation helpers for native integer arrays:

```dart
// Allocates a native int64 array from a Dart List<int>
Pointer<Int64> int64FromList(List<int> values, Arena arena) {
  final ptr = arena.allocate<Int64>(values.length * sizeOf<Int64>());
  for (int i = 0; i < values.length; i++) {
    ptr[i] = values[i];
  }
  return ptr;
}
```

This is used to build shape arrays and int64 label tensors for ORT API calls.

---

## Arena-Based Memory Management

All native memory inside FFI calls is managed via `Arena` from `package:ffi`:

```dart
// Synchronous FFI call
using((arena) {
  final ptr = arena.allocate<Pointer<OrtEnv>>(1);
  _ort.check(_ort.createEnv(0, nullPtr, ptr));
  _env = ptr.value;
  // arena freed automatically when callback exits
});

// Async FFI call
await using((arena) async {
  final inputTensor = _ort.buildFloat32Tensor(
    data: flatInputs, shape: [batchSize, features], arena: arena,
  );
  // ... use tensor
  _ort.releaseValue(inputTensor);
});
```

The arena tracks all allocations made via `arena.allocate<T>(count)` and frees them when the `using()` scope exits — even if an exception is thrown.

---

## Logging Levels

`OrtLoggingLevel` maps to ORT's `OrtLoggingLevel` C enum:

```dart
class OrtLoggingLevel {
  static const int verbose = 0;
  static const int info    = 1;
  static const int warning = 2;
  static const int error   = 3;
  static const int fatal   = 4;
}
```

---

## Tensor Element Types

`OrtTensorElementDataType` mirrors the C enum:

```dart
class OrtTensorElementDataType {
  static const int float32 = 1;
  static const int uint8   = 2;
  static const int int8    = 3;
  static const int uint16  = 4;
  static const int int16   = 5;
  static const int int32   = 6;
  static const int int64   = 7;
  static const int float64 = 11;
  static const int float16 = 10;
}
```

---

## ORT Artifact Directory Layout

The ORT training backend expects this artifact directory structure (produced by `onnxruntime-training-tools`):

```
artifacts/
├── checkpoint           # binary checkpoint (initial parameter values)
├── training_model.onnx  # training graph (forward + backward)
├── eval_model.onnx      # eval graph (forward only)
└── optimizer_model.onnx # Adam optimizer graph
```

Generate with:
```bash
python -m onnxruntime.training.ortmodule.torch_export \
  --model mymodel.pt \
  --output_dir artifacts/
```
