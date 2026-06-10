# Model Export

After training, `ModelExporter` converts the trained session into a deployable model file (`.tflite` or `.onnx`).

---

## Overview

```
TrainingSession
    │
    ▼  ModelExporter.exportToTflite(...)
    │
    ├─ ORT backend ──► exportModelForInferencing (C API)
    │                  → .onnx inference graph
    │                  → optional ONNX graph preprocessing patch
    │                  → quantisation (float16 / int8 / dynamic-range)
    │                  → write to .tflite path
    │
    └─ Dart fallback ► NeuralModel.toJson()
                       → TFLiteConverter.convert() (FlatBuffer JSON)
                       → write to .tflite path
```

---

## ModelExporter

```dart
final exporter = ModelExporter(session: session, spec: spec);

final outputPath = await exporter.exportToTflite(
  outputPath: '/models/classifier_v1.tflite',
  normalizer: normalizer,      // optional: embeds preprocessing
  encoder: encoder,            // optional: embeds preprocessing
  embedPreprocessing: true,    // default true
  quantizationMode: QuantizationMode.float16,  // default
);
print('Saved to $outputPath');
```

### Constructor parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `session` | `TrainingSession` | Trained session |
| `spec` | `ModelSpec` | Model specification (input/output sizes, layers) |

### `exportToTflite` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `outputPath` | `String` | required | Destination file path |
| `normalizer` | `DataNormalizer?` | `null` | Fitted scaler to embed |
| `encoder` | `FeatureEncoder?` | `null` | Fitted encoder to embed |
| `embedPreprocessing` | `bool` | `true` | Patch ONNX graph with normalisation constants |
| `quantizationMode` | `QuantizationMode` | `float16` | Weight quantisation strategy |

---

## QuantizationMode

```dart
enum QuantizationMode {
  none,          // float32 — full precision, largest file
  float16,       // IEEE 754 half-precision — ~50% size reduction
  dynamicRange,  // int8 dynamic-range — ~75% size reduction, no calibration
  int8,          // per-channel static int8 — best compression, needs calibration
}
```

### Size vs accuracy trade-off

| Mode | File size | Accuracy impact | Calibration data needed |
|------|-----------|-----------------|------------------------|
| `none` | 100% | 0% | No |
| `float16` | ~50% | <0.5% | No |
| `dynamicRange` | ~25% | ~1–2% | No |
| `int8` | ~25% | ~1% | Yes (representative dataset) |

---

## ORT Export Path

When using the ORT backend, export calls the C API function `ExportModelForInferencing`:

```c
OrtStatus* ExportModelForInferencing(
  OrtTrainingSession* training_session,
  const char* inference_model_path,
  size_t graph_outputs_len,
  const char* const* graph_output_names
);
```

This strips the training graph (backward pass, gradient accumulators) and writes a clean inference-only ONNX model. The output node name defaults to `"output"`.

---

## Dart Fallback Export Path

For `DartTrainingSession`, export calls `NeuralModel.toJson()` which serialises:

```json
{
  "version": 1,
  "spec": { "inputFeatures": 4, "outputSize": 3, "layerSizes": [4,16,3], "isClassifier": true },
  "layers": [
    {
      "weights": [0.12, -0.34, ...],
      "biases": [0.01, -0.02, 0.00],
      "activation": "relu"
    },
    ...
  ]
}
```

`TFLiteConverter.convert()` then wraps this in a pseudo-TFLite FlatBuffer format compatible with test tooling and the `InferencePipeline` Dart inference path.

---

## TFLiteConverter

For advanced usage, `TFLiteConverter` can be called directly:

```dart
final bytes = TFLiteConverter.convert(
  layers: [
    DenseLayerSpec(
      inSize: 4, outSize: 16,
      weights: List.filled(4 * 16, 0.0),
      biases: List.filled(16, 0.0),
    ),
    DenseLayerSpec(
      inSize: 16, outSize: 3,
      weights: List.filled(16 * 3, 0.0),
      biases: List.filled(3, 0.0),
    ),
  ],
  inputFeatures: 4,
  outputSize: 3,
  isClassifier: true,
);
await File('model.tflite').writeAsBytes(bytes);
```

`DenseLayerSpec.weights` is stored in **row-major order** compatible with TFLite's FullyConnected op (shape `[outSize, inSize]`).

---

## OnnxSerializer

`OnnxSerializer` builds a minimal ONNX `ModelProto` for the Dart-fallback path:

```dart
final onnxBytes = OnnxSerializer.serialize(
  spec: spec,
  layers: layers,
);
```

The resulting graph uses standard ONNX ops: `Gemm`, `Relu`, `Softmax`.

---

## Embedding Preprocessing in the Model

When `embedPreprocessing: true` and a `normalizer` + `encoder` are provided, `ModelExporter` patches the ONNX graph to prepend normalisation `Mul`/`Add` nodes. This makes the deployed model self-contained — callers do not need to run preprocessing separately.

> **Note**: The current implementation returns the ONNX bytes unchanged (placeholder). Full ONNX proto patching is planned for v0.2.0. For now, apply `DataNormalizer.transform()` and `FeatureEncoder.transform()` before calling the model at inference time.

---

## Output Directory Creation

`ModelExporter` automatically creates the output directory:

```dart
await Directory(p.dirname(outputPath)).create(recursive: true);
```

---

## Example: Export with float16 quantisation

```dart
final exporter = ModelExporter(session: session, spec: spec);
final path = await exporter.exportToTflite(
  outputPath: 'models/v1/fraud_detector.tflite',
  normalizer: normalizer,
  encoder: encoder,
  quantizationMode: QuantizationMode.float16,
);
print('Exported $path (${File(path).lengthSync()} bytes)');
```
