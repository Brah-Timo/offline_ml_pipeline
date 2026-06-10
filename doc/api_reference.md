# API Reference

Complete reference for all public types exported by `package:offline_ml_pipeline/offline_ml_pipeline.dart`.

---

## Data Classes

### `CsvLoader`

```dart
class CsvLoader {
  CsvLoader({
    required String filePath,
    required String labelColumn,
    String delimiter = ',',
    bool hasHeader = true,
    int skipRows = 0,
  });

  Future<RawDataset> load();
}
```

### `RawDataset`

```dart
class RawDataset {
  final List<String> featureNames;
  final String labelColumn;
  final List<Map<String, dynamic>> rows;
  int get rowCount;
}
```

### `ProcessedDataset`

```dart
class ProcessedDataset {
  final List<double> allFeatures;     // flat row-major
  final List<dynamic> allLabels;
  final int rowCount;
  final int featureCount;

  DataBatch slice(int start, int end);
  void shuffle({int? seed});
  TrainTestSplit trainTestSplit({
    required double testFraction,
    int? seed,
  });
}
```

### `DataBatch`

```dart
class DataBatch {
  final List<double> features;
  final List<dynamic> labels;
  final int size;
}
```

### `TrainTestSplit`

```dart
class TrainTestSplit {
  final ProcessedDataset train;
  final ProcessedDataset test;
}
```

### `DataNormalizer`

```dart
class DataNormalizer {
  // Factories
  static DataNormalizer fitZScore(RawDataset dataset);
  static DataNormalizer fitMinMax(RawDataset dataset,
      {double min = 0.0, double max = 1.0});

  // Transform
  ProcessedDataset transform(RawDataset dataset);
  ProcessedDataset inverseTransform(ProcessedDataset dataset);

  // Serialisation
  Map<String, dynamic> toJson();
  static DataNormalizer fromJson(Map<String, dynamic> json);

  // Inspect
  Map<String, double> get means;
  Map<String, double> get stdDevs;
  Map<String, double> get mins;
  Map<String, double> get maxs;
}
```

### `FeatureEncoder`

```dart
class FeatureEncoder {
  // Factories
  static FeatureEncoder fitOneHot(RawDataset dataset);
  static FeatureEncoder fitLabel(RawDataset dataset);

  // Transform
  RawDataset transform(RawDataset dataset);
  RawDataset inverseTransform(RawDataset encoded);

  // Inspect
  Map<String, List<String>> get categories;
  List<String> get encodedFeatureNames;

  // Serialisation
  Map<String, dynamic> toJson();
  static FeatureEncoder fromJson(Map<String, dynamic> json);
}
```

---

## Model Classes

### `ModelSpec`

```dart
class ModelSpec {
  final int inputFeatures;
  final int outputSize;
  final List<int> layerSizes;   // [input, hidden..., output]
  final bool isClassifier;

  const ModelSpec({
    required this.inputFeatures,
    required this.outputSize,
    required this.layerSizes,
    required this.isClassifier,
  });
}
```

### `ModelType` (enum)

```dart
enum ModelType { classifier, regressor }
```

### `NeuralModel`

```dart
class NeuralModel {
  NeuralModel(ModelSpec spec);

  double trainStepAdam(
    List<double> flatInputs,
    List<dynamic> labels,
    int batchSize,
    double lr,
  );

  List<List<double>> predictBatch(List<double> flatInputs, int n);

  Map<String, dynamic> toJson();
  static NeuralModel fromJson(Map<String, dynamic> json);
}
```

---

## Training Classes

### `TrainingBackend` (enum)

```dart
enum TrainingBackend { ort, dartFallback }
```

### `TrainingSession` (abstract)

```dart
abstract class TrainingSession {
  final ModelSpec spec;
  final OptimizerConfig optimizerConfig;
  final LossFunction lossFunction;
  TrainingBackend get backend;

  static Future<TrainingSession> create({
    required ModelSpec spec,
    required String artifactDir,
    required OptimizerConfig optimizerConfig,
    required LossFunction lossFunction,
  });

  Future<double> trainStep({
    required List<double> flatInputs,
    required List<dynamic> labels,
    required int batchSize,
  });

  Future<double> evalStep({
    required List<double> flatInputs,
    required List<dynamic> labels,
    required int batchSize,
  });

  Future<List<List<double>>> predict({
    required List<double> flatInputs,
    required int n,
  });

  Future<String> exportInferenceModel(String outputPath);
  Future<void> dispose();
}
```

### `DartTrainingSession`

Concrete subclass of `TrainingSession` using pure-Dart MLP.

### `OptimizerConfig`

```dart
class OptimizerConfig {
  const OptimizerConfig({
    required OptimizerType type,
    required double learningRate,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double epsilon = 1e-8,
    double momentum = 0.9,
    double weightDecay = 0.0,
  });
}
```

### `OptimizerType` (enum)

```dart
enum OptimizerType { adam, sgd, rmsprop }
```

### `LossFunction` (enum)

```dart
enum LossFunction {
  crossEntropy,
  binaryCrossEntropy,
  mse,
  mae,
  huber,
}
```

### `LossFunctions`

```dart
class LossFunctions {
  static double crossEntropy(List<List<double>> preds, List<int> labels);
  static double binaryCrossEntropy(List<double> preds, List<int> labels);
  static double mse(List<double> preds, List<double> labels);
  static double mae(List<double> preds, List<double> labels);
  static double mape(List<double> preds, List<double> labels);
  static double rSquared(List<double> preds, List<double> labels);

  static double compute({
    required LossFunction lossFunction,
    required List<List<double>> preds,
    required List<dynamic> labels,
  });
}
```

### `TrainingLoop`

```dart
class TrainingLoop {
  TrainingLoop({
    required TrainingSession session,
    required ProcessedDataset trainData,
    required ProcessedDataset valData,
    required int epochs,
    required int batchSize,
    required ProgressNotifier progressNotifier,
    EarlyStopping? earlyStopping,
    bool shuffleEachEpoch = true,
    LrSchedule? lrSchedule,
    double initialLr = 0.001,
  });

  final List<EpochRecord> epochHistory;
  Duration duration;

  Future<TrainingMetrics> run();
}
```

### `EarlyStopping`

```dart
class EarlyStopping {
  const EarlyStopping({
    String monitor = 'val_loss',
    int patience = 10,
    double minDelta = 1e-4,
  });
}
```

### `LrSchedule` (abstract)

```dart
abstract class LrSchedule {
  double compute(double initialLr, int epoch);
}
```

Built-in schedules: `ExponentialDecaySchedule`, `StepDecaySchedule`, `CosineAnnealingSchedule`, `WarmupSchedule`.

### `EpochRecord`

```dart
class EpochRecord {
  final int epoch;
  final double trainLoss;
  final double valLoss;
  final DateTime timestamp;
}
```

---

## Metrics Classes

### `TrainingMetrics`

```dart
class TrainingMetrics {
  // Classification
  final double? accuracy;
  final double? f1Score;
  final double? precision;
  final double? recall;
  final List<List<int>>? confusionMatrix;

  // Regression
  final double? mse;
  final double? rmse;
  final double? mae;
  final double? rSquared;
  final double? mape;

  // Common
  final double finalTrainLoss;
  final double finalValLoss;
  final int totalEpochs;
  final bool stoppedEarly;

  String toReport(ModelType modelType);
}
```

### `MetricsCalculator`

```dart
class MetricsCalculator {
  static ClassificationMetrics classification({
    required List<int> preds,
    required List<int> labels,
    required int numClasses,
  });

  static RegressionMetrics regression({
    required List<double> preds,
    required List<double> labels,
  });
}
```

---

## Export Classes

### `QuantizationMode` (enum)

```dart
enum QuantizationMode { none, float16, dynamicRange, int8 }
```

### `ModelExporter`

```dart
class ModelExporter {
  ModelExporter({
    required TrainingSession session,
    required ModelSpec spec,
  });

  Future<String> exportToTflite({
    required String outputPath,
    DataNormalizer? normalizer,
    FeatureEncoder? encoder,
    bool embedPreprocessing = true,
    QuantizationMode quantizationMode = QuantizationMode.float16,
  });
}
```

### `DenseLayerSpec`

```dart
class DenseLayerSpec {
  const DenseLayerSpec({
    required int inSize,
    required int outSize,
    required List<double> weights,  // [outSize × inSize] row-major
    required List<double> biases,   // [outSize]
  });
}
```

### `TFLiteConverter`

```dart
class TFLiteConverter {
  static Uint8List convert({
    required List<DenseLayerSpec> layers,
    required int inputFeatures,
    required int outputSize,
    required bool isClassifier,
  });
}
```

---

## Progress & Error Classes

### `ProgressNotifier`

```dart
class ProgressNotifier {
  Stream<TrainingProgress> get stream;
  Stream<int> get earlyStopStream;     // emits epoch when stopped early
  void notify(TrainingProgress p);
  void notifyEarlyStopped(int epoch);
  Future<void> dispose();
}
```

### `TrainingProgress`

```dart
class TrainingProgress {
  final int epoch;
  final int totalEpochs;
  final double trainLoss;
  final double valLoss;
  final int percentage;
}
```

### Exceptions

```dart
class PipelineException implements Exception { final String message; }
class DataLoadException extends PipelineException { final String filePath; }
class DataPreprocessException extends PipelineException {}
class TrainingStepException extends PipelineException {
  final int epoch;
  final int step;
}
class ModelExportException extends PipelineException { final String outputPath; }
```
