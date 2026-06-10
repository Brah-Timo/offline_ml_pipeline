# Architecture

## Overview

`offline_ml_pipeline` is structured as a layered architecture that keeps the training pipeline, model management, data processing, and native FFI bindings cleanly separated.

```
┌─────────────────────────────────────────────────┐
│                  Public API                      │
│  offline_ml_pipeline.dart (barrel export)        │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │   Training Layer       │
        │  TrainingLoop          │
        │  TrainingSession       │
        │  TrainingMetrics       │
        │  OptimizerConfig       │
        │  LossFunctions         │
        └───────────┬───────────┘
                    │
     ┌──────────────┼──────────────┐
     │              │              │
┌────▼────┐  ┌──────▼──────┐  ┌───▼────────┐
│  ORT    │  │ Dart Fallback│  │  Export    │
│ Backend │  │  (NeuralModel│  │  Layer     │
│ (FFI)   │  │   + Adam)    │  │            │
└────┬────┘  └─────────────┘  └───┬────────┘
     │                             │
┌────▼─────────────────────────────▼────────┐
│              Data Layer                    │
│  CsvLoader  DataNormalizer  FeatureEncoder │
│  DataSchema ProcessedDataset              │
└───────────────────────────────────────────┘
```

---

## Package Structure

```
lib/
├── offline_ml_pipeline.dart          # Public barrel export
└── src/
    ├── data/
    │   ├── csv_loader.dart           # CSV → RawDataset
    │   ├── data_normalizer.dart      # Z-score / min-max scaling
    │   ├── data_schema.dart          # RawDataset, ProcessedDataset
    │   └── feature_encoder.dart      # One-hot / label encoding
    ├── export/
    │   ├── model_exporter.dart       # High-level export coordinator
    │   ├── onnx_serializer.dart      # ONNX graph builder (Dart fallback)
    │   └── tflite_converter.dart     # ONNX → TFLite FlatBuffer converter
    ├── ffi/
    │   ├── native_memory.dart        # Arena helpers, int64 allocation
    │   ├── ort_bindings.dart         # OrtBindings singleton (dlopen)
    │   └── ort_types.dart            # Opaque FFI structs + typedef pairs
    ├── models/
    │   ├── base_model.dart           # ModelSpec, abstract interfaces
    │   ├── model_type.dart           # ModelType enum
    │   └── neural_model.dart         # Pure-Dart MLP with Adam optimizer
    ├── training/
    │   ├── loss_functions.dart       # CrossEntropy, MSE, R², MAPE
    │   ├── optimizer_config.dart     # OptimizerConfig, LrSchedule
    │   ├── training_loop.dart        # Epoch/batch loop, early stopping
    │   ├── training_metrics.dart     # MetricsCalculator, TrainingMetrics
    │   └── training_session.dart     # Abstract TrainingSession + backends
    └── utils/
        ├── error_handler.dart        # Typed exceptions
        └── progress_notifier.dart    # Stream-based progress events
test/
├── integration/
│   └── full_pipeline_test.dart      # End-to-end CSV → train → export
└── unit/
    ├── csv_loader_test.dart
    ├── data_normalizer_test.dart
    └── training_metrics_test.dart
```

---

## Key Design Decisions

### 1. Backend abstraction via `TrainingSession`

`TrainingSession` is an abstract class with two concrete subclasses:

- `_OrtBackendSession` — wraps the ONNX Runtime On-Device Training C API via `dart:ffi`. Selected when the native library is available and `OrtBindings.instance` succeeds.
- `DartTrainingSession` — pure-Dart MLP using the Adam optimizer implemented in `NeuralModel`. Zero native dependencies; works on all platforms.

The factory `TrainingSession.create(...)` tries ORT first and silently falls back to Dart if the `.so`/`.dylib` is missing.

### 2. Dart FFI opaque types

All ONNX Runtime C API handle types (`OrtEnv`, `OrtTrainingSession`, `OrtCheckpointState`, etc.) are modelled as `final class Foo extends Opaque {}` in `ort_types.dart`. This satisfies Dart's `Pointer<T extends NativeType>` constraint.

### 3. Arena-based native memory

All native allocations inside FFI calls use `package:ffi`'s `Arena` allocator (via `using()` / `await using()`). The arena automatically frees all C memory when the callback exits — no manual `free()` calls needed.

### 4. Quantization pipeline

`QuantizationMode` controls how the exported ONNX/TFLite model is quantized:

| Mode | Description |
|------|-------------|
| `none` | Float32 weights — full precision |
| `float16` | IEEE 754 half-precision — halves model size |
| `dynamicRange` | Int8 dynamic-range quantization |
| `int8` | Per-channel static int8 (requires calibration) |

### 5. Progress events

`ProgressNotifier` wraps a `StreamController<TrainingProgress>`. The caller subscribes to `notifier.stream` and receives one event per epoch. This keeps the training loop UI-agnostic.

---

## Data Flow

```
CSV file
  │
  ▼ CsvLoader.load()
RawDataset (List<Map<String,String>>)
  │
  ▼ DataNormalizer.fitZScore(raw).transform(raw)
  ▼ FeatureEncoder.fitOneHot(raw).transform(raw)
ProcessedDataset (features: List<double>, labels: List<dynamic>)
  │
  ├─ trainTestSplit()
  │   ├─ split.train → TrainingLoop
  │   └─ split.test  → TrainingLoop (validation)
  │
  ▼ TrainingLoop.run()
TrainingMetrics
  │
  ▼ ModelExporter.exportToTflite()
.tflite / .onnx file
```

---

## Error Hierarchy

```
PipelineException
├── DataLoadException       — CSV parse errors
├── DataPreprocessException — normalisation / encoding errors
├── TrainingStepException   — NaN/Inf loss, ORT status errors
└── ModelExportException    — file I/O, ORT export errors
```

All exceptions carry contextual fields (file path, epoch, step) for precise debugging.
