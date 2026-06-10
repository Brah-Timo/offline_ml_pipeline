// ignore_for_file: lines_longer_than_80_chars

import 'dart:math' as math;
import 'base_model.dart';

// ─────────────────────────────────────────────────────────────────────────────
// NeuralModel — pure-Dart MLP (fallback when ORT is unavailable)
// ─────────────────────────────────────────────────────────────────────────────

/// A pure-Dart multi-layer perceptron.
///
/// Used as the **pure-Dart fallback** when ONNX Runtime native bindings
/// are not yet available on a platform, or for unit-testing the training
/// loop without native code.
///
/// Supports:
/// - ReLU hidden activations
/// - Softmax output (classifier) / linear output (regressor)
/// - Mini-batch SGD / Adam weight update
/// - Forward pass, loss computation, and numeric gradient (finite differences)
class NeuralModel {
  final ModelSpec spec;
  late WeightStore weights;

  // Adam optimiser state
  late WeightStore _m; // first moment
  late WeightStore _v; // second moment
  int _t = 0;          // step counter

  static const double _beta1 = 0.9;
  static const double _beta2 = 0.999;
  static const double _eps = 1e-8;

  NeuralModel(this.spec) {
    weights = WeightStore.heInit(spec.layerSizes);
    _m = WeightStore.zeros(spec.layerSizes);
    _v = WeightStore.zeros(spec.layerSizes);
  }

  // ── Forward pass ───────────────────────────────────────────────────────

  /// Returns raw logits for classifier or a scalar list for regressor.
  List<double> forward(List<double> input) {
    var activation = input;
    final lastLayer = weights.layers.length - 1;

    for (int li = 0; li < weights.layers.length; li++) {
      final layer = weights.layers[li];
      final z = _dense(activation, layer);

      if (li < lastLayer) {
        activation = z.map(_relu).toList();
      } else {
        // Output layer
        activation = spec.isClassifier ? _softmax(z) : z;
      }
    }

    return activation;
  }

  /// Predicts a batch and returns one output vector per sample.
  List<List<double>> predictBatch(List<double> flatInputs, int batchSize) {
    final feats = spec.inputFeatures;
    final out = <List<double>>[];
    for (int i = 0; i < batchSize; i++) {
      out.add(forward(flatInputs.sublist(i * feats, (i + 1) * feats)));
    }
    return out;
  }

  // ── Loss computation ───────────────────────────────────────────────────

  /// Cross-entropy loss for classifier.
  double crossEntropyLoss(List<List<double>> preds, List<int> labels) {
    double loss = 0;
    for (int i = 0; i < preds.length; i++) {
      final prob = preds[i][labels[i]].clamp(1e-9, 1.0 - 1e-9);
      loss -= math.log(prob);
    }
    return loss / preds.length;
  }

  /// Mean squared error for regressor.
  double mseLoss(List<List<double>> preds, List<double> targets) {
    double loss = 0;
    for (int i = 0; i < preds.length; i++) {
      final diff = preds[i][0] - targets[i];
      loss += diff * diff;
    }
    return loss / preds.length;
  }

  // ── Weight update (Adam) ───────────────────────────────────────────────

  /// Computes numeric gradients via central finite differences and applies
  /// an Adam update step.
  ///
  /// ε for finite difference is 1e-4. Suitable for small models / few features.
  ///
  /// Returns the loss **before** the update.
  double trainStepAdam(
    List<double> flatInputs,
    List<dynamic> labels,
    int batchSize,
    double learningRate,
  ) {
    _t++;

    // ── Compute loss (forward)
    final preds = predictBatch(flatInputs, batchSize);
    final double loss;
    if (spec.isClassifier) {
      loss = crossEntropyLoss(
          preds, labels.map((l) => (l as num).toInt()).toList());
    } else {
      loss = mseLoss(
          preds, labels.map((l) => (l as num).toDouble()).toList());
    }

    // ── Numeric gradient for each parameter
    const h = 1e-4;

    for (int li = 0; li < weights.layers.length; li++) {
      final layer = weights.layers[li];
      final mLayer = _m.layers[li];
      final vLayer = _v.layers[li];

      // Weights
      for (int wi = 0; wi < layer.weights.length; wi++) {
        final orig = layer.weights[wi];

        layer.weights[wi] = orig + h;
        final lossPlus = _computeLoss(flatInputs, labels, batchSize);

        layer.weights[wi] = orig - h;
        final lossMinus = _computeLoss(flatInputs, labels, batchSize);

        layer.weights[wi] = orig;
        final grad = (lossPlus - lossMinus) / (2 * h);

        // Adam update
        mLayer.weights[wi] = _beta1 * mLayer.weights[wi] + (1 - _beta1) * grad;
        vLayer.weights[wi] = _beta2 * vLayer.weights[wi] + (1 - _beta2) * grad * grad;

        final mHat = mLayer.weights[wi] / (1 - math.pow(_beta1, _t));
        final vHat = vLayer.weights[wi] / (1 - math.pow(_beta2, _t));

        layer.weights[wi] -= learningRate * mHat / (math.sqrt(vHat) + _eps);
      }

      // Biases
      for (int bi = 0; bi < layer.biases.length; bi++) {
        final orig = layer.biases[bi];

        layer.biases[bi] = orig + h;
        final lossPlus = _computeLoss(flatInputs, labels, batchSize);

        layer.biases[bi] = orig - h;
        final lossMinus = _computeLoss(flatInputs, labels, batchSize);

        layer.biases[bi] = orig;
        final grad = (lossPlus - lossMinus) / (2 * h);

        mLayer.biases[bi] = _beta1 * mLayer.biases[bi] + (1 - _beta1) * grad;
        vLayer.biases[bi] = _beta2 * vLayer.biases[bi] + (1 - _beta2) * grad * grad;

        final mHat = mLayer.biases[bi] / (1 - math.pow(_beta1, _t));
        final vHat = vLayer.biases[bi] / (1 - math.pow(_beta2, _t));

        layer.biases[bi] -= learningRate * mHat / (math.sqrt(vHat) + _eps);
      }
    }

    return loss;
  }

  // ── Internal helpers ───────────────────────────────────────────────────

  double _computeLoss(
    List<double> flatInputs,
    List<dynamic> labels,
    int batchSize,
  ) {
    final preds = predictBatch(flatInputs, batchSize);
    if (spec.isClassifier) {
      return crossEntropyLoss(
          preds, labels.map((l) => (l as num).toInt()).toList());
    } else {
      return mseLoss(
          preds, labels.map((l) => (l as num).toDouble()).toList());
    }
  }

  List<double> _dense(List<double> input, LayerWeights layer) {
    final out = List<double>.filled(layer.outSize, 0.0);
    for (int j = 0; j < layer.outSize; j++) {
      double sum = layer.biases[j];
      for (int i = 0; i < layer.inSize; i++) {
        sum += input[i] * layer.weights[i * layer.outSize + j];
      }
      out[j] = sum;
    }
    return out;
  }

  double _relu(double x) => x > 0.0 ? x : 0.0;

  List<double> _softmax(List<double> z) {
    final maxZ = z.reduce(math.max);
    final exps = z.map((v) => math.exp(v - maxZ)).toList();
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((v) => v / sum).toList();
  }

  // ── Serialisation ──────────────────────────────────────────────────────

  Map<String, dynamic> toJson() => {
        'spec': spec.toJson(),
        'weights': weights.toJson(),
        't': _t,
      };

  factory NeuralModel.fromJson(Map<String, dynamic> j) {
    final model = NeuralModel(ModelSpec.fromJson(j['spec'] as Map<String, dynamic>));
    model.weights = WeightStore.fromJson(j['weights'] as Map<String, dynamic>);
    model._t = j['t'] as int;
    return model;
  }
}
