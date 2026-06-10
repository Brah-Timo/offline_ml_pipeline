# Getting Started with offline_ml_pipeline

`offline_ml_pipeline` is a Flutter/Dart package for **on-device machine learning training and inference**.  
It supports two backends:

| Backend | Description | Platform requirement |
|---------|-------------|----------------------|
| **ORT** | ONNX Runtime On-Device Training C API | Requires native `.so`/`.dylib`/`.dll` |
| **Dart fallback** | Pure-Dart MLP with Adam optimizer | Any platform, no native deps |

---

## Installation

Add the package to your `pubspec.yaml`:

```yaml
dependencies:
  offline_ml_pipeline: ^0.1.0
```

Then run:

```bash
flutter pub get
```

---

## Quick-Start Example

```dart
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';

Future<void> main() async {
  // 1. Define the model
  final spec = ModelSpec(
    inputFeatures: 4,
    outputSize: 3,
    layerSizes: [4, 16, 8, 3],
    isClassifier: true,
  );

  // 2. Configure the optimizer
  const optimizer = OptimizerConfig(
    type: OptimizerType.adam,
    learningRate: 0.001,
    beta1: 0.9,
    beta2: 0.999,
  );

  // 3. Create a training session (auto-selects ORT or Dart fallback)
  final session = await TrainingSession.create(
    spec: spec,
    artifactDir: '/path/to/ort_artifacts',
    optimizerConfig: optimizer,
    lossFunction: LossFunction.crossEntropy,
  );

  // 4. Load and preprocess data
  final loader = CsvLoader(filePath: 'assets/data.csv', labelColumn: 'label');
  final raw = await loader.load();

  final normalizer = DataNormalizer.fitZScore(raw);
  final encoder = FeatureEncoder.fitOneHot(raw);
  final processed = normalizer.transform(encoder.transform(raw));

  final split = processed.trainTestSplit(testFraction: 0.2);

  // 5. Run the training loop
  final notifier = ProgressNotifier();
  notifier.stream.listen((progress) {
    print('Epoch ${progress.epoch}/${progress.totalEpochs} '
          '— loss: ${progress.trainLoss.toStringAsFixed(4)}');
  });

  final loop = TrainingLoop(
    session: session,
    trainData: split.train,
    valData: split.test,
    epochs: 50,
    batchSize: 32,
    progressNotifier: notifier,
    earlyStopping: const EarlyStopping(patience: 5),
  );

  final metrics = await loop.run();
  print(metrics.toReport(ModelType.classifier));

  // 6. Export model
  final exporter = ModelExporter(session: session, spec: spec);
  final path = await exporter.exportToTflite(
    outputPath: '/output/model.tflite',
    normalizer: normalizer,
    encoder: encoder,
  );
  print('Model exported to $path');

  await session.dispose();
}
```

---

## Step-by-Step Guide

### Step 1 — Define a ModelSpec

```dart
final spec = ModelSpec(
  inputFeatures: 10,   // number of input columns after encoding
  outputSize: 2,       // number of classes (classifier) or 1 (regressor)
  layerSizes: [10, 32, 16, 2],
  isClassifier: true,
);
```

`layerSizes` describes the full network depth including input and output.  
Each adjacent pair defines a `DenseLayer`: `[10→32]`, `[32→16]`, `[16→2]`.

### Step 2 — Configure the Optimizer

```dart
const config = OptimizerConfig(
  type: OptimizerType.adam,
  learningRate: 1e-3,
  weightDecay: 1e-4,   // optional L2 regularisation
);
```

Supported types: `adam`, `sgd`, `rmsprop`.

### Step 3 — Load CSV Data

```dart
final loader = CsvLoader(
  filePath: 'assets/iris.csv',
  labelColumn: 'species',
  delimiter: ',',
  hasHeader: true,
);
final dataset = await loader.load();
```

### Step 4 — Preprocess

```dart
final normalizer = DataNormalizer.fitZScore(dataset);
final encoder    = FeatureEncoder.fitOneHot(dataset);
final processed  = normalizer.transform(encoder.transform(dataset));
final split      = processed.trainTestSplit(testFraction: 0.2, seed: 42);
```

### Step 5 — Train

```dart
final metrics = await TrainingLoop(
  session: session,
  trainData: split.train,
  valData: split.test,
  epochs: 100,
  batchSize: 32,
  progressNotifier: ProgressNotifier(),
  lrSchedule: ExponentialDecaySchedule(decayRate: 0.95),
).run();
```

### Step 6 — Export

```dart
final exporter = ModelExporter(session: session, spec: spec);
await exporter.exportToTflite(
  outputPath: 'model.tflite',
  quantizationMode: QuantizationMode.float16,
);
```

---

## Next Steps

- [Architecture overview](architecture.md)
- [Data pipeline details](data_pipeline.md)
- [Training loop deep-dive](training.md)
- [Export formats](export.md)
- [FFI bindings](ffi_bindings.md)
- [API reference](api_reference.md)
- [Performance guide](performance.md)
- [Testing](testing.md)
- [Contributing](contributing.md)
