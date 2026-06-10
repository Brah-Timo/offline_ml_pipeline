// ignore_for_file: lines_longer_than_80_chars

import 'model_type.dart';

// ─────────────────────────────────────────────────────────────────────────────
// ModelSpec
// ─────────────────────────────────────────────────────────────────────────────

/// Complete specification of the neural network to be built.
///
/// Created from [PipelineConfig] after the dataset schema is known.
/// Passed to the artifact generator and the ORT training session.
class ModelSpec {
  /// Type of supervised task.
  final ModelType modelType;

  /// Number of input features (after encoding + normalisation).
  final int inputFeatures;

  /// For classifiers: number of output classes.
  /// For regressors: always 1.
  final int outputSize;

  /// Which architecture to instantiate.
  final ModelArchitecture architecture;

  /// Dropout rate applied to hidden layers during training.
  final double dropoutRate;

  /// Whether to add BatchNorm layers after hidden layers.
  final bool useBatchNorm;

  const ModelSpec({
    required this.modelType,
    required this.inputFeatures,
    required this.outputSize,
    this.architecture = ModelArchitecture.mlpShallow,
    this.dropoutRate = 0.2,
    this.useBatchNorm = true,
  });

  // ── Derived helpers ────────────────────────────────────────────────────

  /// True for classification tasks.
  bool get isClassifier => modelType == ModelType.classifier;

  /// True for regression tasks.
  bool get isRegressor => modelType == ModelType.regressor;

  /// Returns layer sizes as a list `[input, hidden…, output]`.
  List<int> get layerSizes {
    switch (architecture) {
      case ModelArchitecture.linear:
        return [inputFeatures, outputSize];
      case ModelArchitecture.mlpShallow:
        return [inputFeatures, 64, outputSize];
      case ModelArchitecture.mlpDeep:
        return [inputFeatures, 128, 64, outputSize];
    }
  }

  /// Human-readable summary.
  @override
  String toString() => 'ModelSpec('
      'type: ${modelType.name}, '
      'in: $inputFeatures, '
      'out: $outputSize, '
      'arch: ${architecture.name}, '
      'layers: $layerSizes)';

  Map<String, dynamic> toJson() => {
        'modelType': modelType.name,
        'inputFeatures': inputFeatures,
        'outputSize': outputSize,
        'architecture': architecture.name,
        'dropoutRate': dropoutRate,
        'useBatchNorm': useBatchNorm,
      };

  factory ModelSpec.fromJson(Map<String, dynamic> j) => ModelSpec(
        modelType: ModelType.values.firstWhere((e) => e.name == j['modelType']),
        inputFeatures: j['inputFeatures'] as int,
        outputSize: j['outputSize'] as int,
        architecture: ModelArchitecture.values.firstWhere(
          (e) => e.name == j['architecture'],
        ),
        dropoutRate: (j['dropoutRate'] as num).toDouble(),
        useBatchNorm: j['useBatchNorm'] as bool,
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// WeightStore
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight in-memory weight container used by the pure-Dart fallback
/// trainer (for platforms where ORT is not yet available).
///
/// Weights are stored as flat [Float32List]-compatible lists.
class WeightStore {
  final List<LayerWeights> layers;

  const WeightStore(this.layers);

  factory WeightStore.zeros(List<int> layerSizes) {
    final layers = <LayerWeights>[];
    for (int i = 0; i < layerSizes.length - 1; i++) {
      final inSize = layerSizes[i];
      final outSize = layerSizes[i + 1];
      layers.add(LayerWeights.zeros(inSize, outSize));
    }
    return WeightStore(layers);
  }

  factory WeightStore.heInit(List<int> layerSizes, {int seed = 42}) {
    final layers = <LayerWeights>[];
    var rngState = seed;

    for (int i = 0; i < layerSizes.length - 1; i++) {
      final inSize = layerSizes[i];
      final outSize = layerSizes[i + 1];
      layers.add(LayerWeights.heInit(inSize, outSize, rngState));
      rngState += outSize;
    }
    return WeightStore(layers);
  }

  Map<String, dynamic> toJson() => {
        'layers': layers.map((l) => l.toJson()).toList(),
      };

  factory WeightStore.fromJson(Map<String, dynamic> j) => WeightStore(
        (j['layers'] as List)
            .map((l) => LayerWeights.fromJson(l as Map<String, dynamic>))
            .toList(),
      );
}

/// Weights and biases for a single dense layer.
class LayerWeights {
  /// Shape: [inSize × outSize] (row-major).
  final List<double> weights;

  /// Shape: [outSize].
  final List<double> biases;

  final int inSize;
  final int outSize;

  LayerWeights({
    required this.weights,
    required this.biases,
    required this.inSize,
    required this.outSize,
  });

  factory LayerWeights.zeros(int inSize, int outSize) => LayerWeights(
        weights: List.filled(inSize * outSize, 0.0),
        biases: List.filled(outSize, 0.0),
        inSize: inSize,
        outSize: outSize,
      );

  factory LayerWeights.heInit(int inSize, int outSize, int seed) {
    // He initialisation: w ~ N(0, sqrt(2/inSize))
    final scale = _sqrt(2.0 / inSize);
    var state = seed;
    double nextRand() {
      // Box-Muller with LCG
      state = (state * 1664525 + 1013904223) & 0xFFFFFFFF;
      final u1 = (state & 0xFFFF) / 65535.0;
      state = (state * 1664525 + 1013904223) & 0xFFFFFFFF;
      final u2 = (state & 0xFFFF) / 65535.0;
      final r = _sqrt(-2.0 * _ln(u1 + 1e-9));
      final theta = 6.2831853 * u2;
      return r * _cos(theta) * scale;
    }

    return LayerWeights(
      weights: List.generate(inSize * outSize, (_) => nextRand()),
      biases: List.filled(outSize, 0.0),
      inSize: inSize,
      outSize: outSize,
    );
  }

  Map<String, dynamic> toJson() => {
        'weights': weights,
        'biases': biases,
        'inSize': inSize,
        'outSize': outSize,
      };

  factory LayerWeights.fromJson(Map<String, dynamic> j) => LayerWeights(
        weights: List<double>.from(
            (j['weights'] as List).map((v) => (v as num).toDouble())),
        biases: List<double>.from(
            (j['biases'] as List).map((v) => (v as num).toDouble())),
        inSize: j['inSize'] as int,
        outSize: j['outSize'] as int,
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure-Dart math helpers (avoid dart:math import in this file)
// ─────────────────────────────────────────────────────────────────────────────
double _sqrt(double x) {
  if (x <= 0) return 0;
  double g = x;
  for (int i = 0; i < 40; i++) {
    g = (g + x / g) * 0.5;
  }
  return g;
}

double _ln(double x) {
  if (x <= 0) return double.negativeInfinity;
  double sum = 0;
  double t = (x - 1) / (x + 1);
  double tPow = t;
  for (int i = 1; i <= 60; i += 2) {
    sum += tPow / i;
    tPow *= t * t;
  }
  return 2 * sum;
}

double _cos(double x) {
  // Taylor: cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720 + …
  double r = 1;
  double term = 1;
  for (int k = 1; k <= 10; k++) {
    term *= -x * x / ((2 * k - 1) * (2 * k));
    r += term;
  }
  return r;
}
