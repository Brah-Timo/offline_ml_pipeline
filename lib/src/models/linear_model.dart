// ignore_for_file: lines_longer_than_80_chars

import 'dart:math' as math;
import 'base_model.dart';

// ─────────────────────────────────────────────────────────────────────────────
// LinearModel — Logistic / Linear Regression (pure Dart)
// ─────────────────────────────────────────────────────────────────────────────

/// Pure-Dart logistic regression (classifier) or linear regression (regressor).
///
/// Fastest to train; good as a baseline or when data is roughly linearly
/// separable. Uses mini-batch SGD with optional L2 regularisation.
class LinearModel {
  final ModelSpec spec;

  // Parameters: [inputFeatures × outputSize] weight matrix + [outputSize] bias
  late List<double> _W; // shape: inputFeatures * outputSize
  late List<double> _b; // shape: outputSize

  LinearModel(this.spec) {
    _initParams();
  }

  // ── Initialisation ─────────────────────────────────────────────────────

  void _initParams() {
    final scale = 0.01;
    final rng = math.Random(42);
    _W = List.generate(
      spec.inputFeatures * spec.outputSize,
      (_) => (rng.nextDouble() - 0.5) * scale,
    );
    _b = List.filled(spec.outputSize, 0.0);
  }

  // ── Forward pass ───────────────────────────────────────────────────────

  /// Returns class probabilities (classifier) or scalar prediction (regressor).
  List<double> predict(List<double> input) {
    final z = _matVecMul(input);
    return spec.isClassifier ? _softmax(z) : z;
  }

  List<List<double>> predictBatch(List<double> flat, int n) {
    final f = spec.inputFeatures;
    return List.generate(n, (i) => predict(flat.sublist(i * f, (i + 1) * f)));
  }

  // ── Training step (SGD) ────────────────────────────────────────────────

  /// Runs one SGD step on a mini-batch.
  /// Returns loss value before the update.
  double trainStep({
    required List<double> flatInputs,
    required List<dynamic> labels,
    required int batchSize,
    required double learningRate,
    double l2Lambda = 1e-4,
  }) {
    final f = spec.inputFeatures;
    final o = spec.outputSize;
    final n = batchSize;

    // ── Accumulate gradients
    final dW = List<double>.filled(f * o, 0.0);
    final db = List<double>.filled(o, 0.0);
    double totalLoss = 0.0;

    for (int i = 0; i < n; i++) {
      final x = flatInputs.sublist(i * f, (i + 1) * f);
      final z = _matVecMul(x);
      final out = spec.isClassifier ? _softmax(z) : z;

      List<double> delta;
      if (spec.isClassifier) {
        final label = labels[i] as int;
        totalLoss -= math.log(out[label].clamp(1e-9, 1.0));
        // delta = p - one_hot(y)
        delta = List<double>.from(out);
        delta[label] -= 1.0;
      } else {
        final target = (labels[i] as num).toDouble();
        final diff = out[0] - target;
        totalLoss += diff * diff;
        delta = [diff];
      }

      // dW += x^T * delta
      for (int fi = 0; fi < f; fi++) {
        for (int oi = 0; oi < o; oi++) {
          dW[fi * o + oi] += x[fi] * delta[oi];
        }
      }
      for (int oi = 0; oi < o; oi++) {
        db[oi] += delta[oi];
      }
    }

    final lr = learningRate / n;

    // ── Apply gradient update + L2 regularisation
    for (int k = 0; k < _W.length; k++) {
      _W[k] -= lr * (dW[k] + l2Lambda * _W[k]);
    }
    for (int k = 0; k < _b.length; k++) {
      _b[k] -= lr * db[k];
    }

    return spec.isClassifier
        ? totalLoss / n
        : totalLoss / n;
  }

  // ── Internal math ──────────────────────────────────────────────────────

  List<double> _matVecMul(List<double> x) {
    final o = spec.outputSize;
    final result = List<double>.filled(o, 0.0);
    for (int oi = 0; oi < o; oi++) {
      double sum = _b[oi];
      for (int fi = 0; fi < spec.inputFeatures; fi++) {
        sum += x[fi] * _W[fi * o + oi];
      }
      result[oi] = sum;
    }
    return result;
  }

  List<double> _softmax(List<double> z) {
    final maxZ = z.reduce(math.max);
    final exps = z.map((v) => math.exp(v - maxZ)).toList();
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((v) => v / sum).toList();
  }

  // ── Serialisation ──────────────────────────────────────────────────────

  Map<String, dynamic> toJson() => {
        'spec': spec.toJson(),
        'W': _W,
        'b': _b,
      };

  factory LinearModel.fromJson(Map<String, dynamic> j) {
    final m = LinearModel(ModelSpec.fromJson(j['spec'] as Map<String, dynamic>));
    m._W = List<double>.from((j['W'] as List).map((v) => (v as num).toDouble()));
    m._b = List<double>.from((j['b'] as List).map((v) => (v as num).toDouble()));
    return m;
  }
}
