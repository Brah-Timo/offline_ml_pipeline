import 'dart:math' as math;

// ─────────────────────────────────────────────────────────────────────────────
// LossFunction enum
// ─────────────────────────────────────────────────────────────────────────────

/// The objective function minimised during training.
enum LossFunction {
  /// Categorical cross-entropy.
  ///
  /// `L = -Σ y_i * log(ŷ_i)` over all classes.
  /// Use for multi-class classification tasks.
  crossEntropy,

  /// Binary cross-entropy.
  ///
  /// `L = -(y*log(ŷ) + (1-y)*log(1-ŷ))`
  /// Use for binary classification tasks.
  binaryCrossEntropy,

  /// Mean Squared Error.
  ///
  /// `L = (1/N) * Σ (y - ŷ)²`
  /// Default for regression.
  mse,

  /// Mean Absolute Error.
  ///
  /// `L = (1/N) * Σ |y - ŷ|`
  /// More robust to outliers than MSE.
  mae,

  /// Huber loss (smooth L1).
  ///
  /// Combines MAE for large errors and MSE for small errors.
  huber,
}

// ─────────────────────────────────────────────────────────────────────────────
// LossFunctions — static computation methods
// ─────────────────────────────────────────────────────────────────────────────

/// Pure-Dart implementations of common loss functions.
///
/// Used by the Dart-fallback trainer (when ORT is unavailable)
/// and by evaluation code.
class LossFunctions {
  LossFunctions._();

  // ── Classification ─────────────────────────────────────────────────────

  /// Categorical cross-entropy over a batch.
  ///
  /// [preds]: batch of probability vectors, shape [N × C].
  /// [labels]: integer class indices, shape [N].
  static double crossEntropy(
    List<List<double>> preds,
    List<int> labels,
  ) {
    assert(preds.length == labels.length);
    double loss = 0.0;
    for (int i = 0; i < preds.length; i++) {
      final prob = preds[i][labels[i]].clamp(1e-9, 1.0 - 1e-9);
      loss -= math.log(prob);
    }
    return loss / preds.length;
  }

  /// Binary cross-entropy over a batch.
  ///
  /// [preds]: batch of scalar probabilities ∈ (0, 1), shape [N].
  /// [labels]: binary labels 0 or 1, shape [N].
  static double binaryCrossEntropy(
    List<double> preds,
    List<int> labels,
  ) {
    assert(preds.length == labels.length);
    double loss = 0.0;
    for (int i = 0; i < preds.length; i++) {
      final p = preds[i].clamp(1e-9, 1.0 - 1e-9);
      final y = labels[i].toDouble();
      loss -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p);
    }
    return loss / preds.length;
  }

  // ── Regression ─────────────────────────────────────────────────────────

  /// Mean Squared Error.
  static double mse(List<double> preds, List<double> targets) {
    assert(preds.length == targets.length);
    double loss = 0.0;
    for (int i = 0; i < preds.length; i++) {
      final diff = preds[i] - targets[i];
      loss += diff * diff;
    }
    return loss / preds.length;
  }

  /// Mean Absolute Error.
  static double mae(List<double> preds, List<double> targets) {
    assert(preds.length == targets.length);
    double loss = 0.0;
    for (int i = 0; i < preds.length; i++) {
      loss += (preds[i] - targets[i]).abs();
    }
    return loss / preds.length;
  }

  /// Huber loss with configurable [delta] (default 1.0).
  static double huber(
    List<double> preds,
    List<double> targets, {
    double delta = 1.0,
  }) {
    assert(preds.length == targets.length);
    double loss = 0.0;
    for (int i = 0; i < preds.length; i++) {
      final diff = (preds[i] - targets[i]).abs();
      if (diff <= delta) {
        loss += 0.5 * diff * diff;
      } else {
        loss += delta * (diff - 0.5 * delta);
      }
    }
    return loss / preds.length;
  }

  // ── Root / derived metrics ─────────────────────────────────────────────

  /// Root Mean Squared Error.
  static double rmse(List<double> preds, List<double> targets) =>
      math.sqrt(mse(preds, targets));

  /// R² coefficient of determination.
  ///
  /// R² = 1 - SS_res / SS_tot
  static double rSquared(List<double> preds, List<double> targets) {
    if (targets.isEmpty) return 0.0;
    final mean = targets.reduce((a, b) => a + b) / targets.length;
    double ssTot = 0, ssRes = 0;
    for (int i = 0; i < targets.length; i++) {
      ssTot += math.pow(targets[i] - mean, 2);
      ssRes += math.pow(targets[i] - preds[i], 2);
    }
    if (ssTot < 1e-9) return 1.0;
    return 1.0 - ssRes / ssTot;
  }

  /// Mean Absolute Percentage Error (values in [0, 1] scale, not percent).
  static double mape(List<double> preds, List<double> targets) {
    assert(preds.length == targets.length);
    double loss = 0.0;
    int count = 0;
    for (int i = 0; i < targets.length; i++) {
      if (targets[i].abs() < 1e-9) continue; // avoid division by zero
      loss += ((targets[i] - preds[i]) / targets[i]).abs();
      count++;
    }
    return count == 0 ? 0.0 : loss / count;
  }

  // ── Dispatch helper ────────────────────────────────────────────────────

  /// Computes the configured [lossFunction] on a batch.
  ///
  /// For classification: [labels] contains integers.
  /// For regression: [labels] contains doubles.
  static double compute({
    required LossFunction lossFunction,
    required List<List<double>> preds,
    required List<dynamic> labels,
  }) {
    switch (lossFunction) {
      case LossFunction.crossEntropy:
        return crossEntropy(
            preds, labels.map((l) => (l as num).toInt()).toList());
      case LossFunction.binaryCrossEntropy:
        return binaryCrossEntropy(
          preds.map((p) => p[0]).toList(),
          labels.map((l) => (l as num).toInt()).toList(),
        );
      case LossFunction.mse:
        return mse(
          preds.map((p) => p[0]).toList(),
          labels.map((l) => (l as num).toDouble()).toList(),
        );
      case LossFunction.mae:
        return mae(
          preds.map((p) => p[0]).toList(),
          labels.map((l) => (l as num).toDouble()).toList(),
        );
      case LossFunction.huber:
        return huber(
          preds.map((p) => p[0]).toList(),
          labels.map((l) => (l as num).toDouble()).toList(),
        );
    }
  }
}
