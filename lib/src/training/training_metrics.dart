// ignore_for_file: lines_longer_than_80_chars

import 'dart:math' as math;
import '../models/model_type.dart';
import 'loss_functions.dart';

// ─────────────────────────────────────────────────────────────────────────────
// EpochRecord
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of training and validation metrics for a single epoch.
class EpochRecord {
  final int epoch;
  final double trainLoss;
  final double valLoss;
  final Map<String, double> extra; // optional (e.g. accuracy, f1)
  final DateTime timestamp;

  const EpochRecord({
    required this.epoch,
    required this.trainLoss,
    required this.valLoss,
    this.extra = const {},
    required this.timestamp,
  });

  Map<String, dynamic> toJson() => {
        'epoch': epoch,
        'trainLoss': trainLoss,
        'valLoss': valLoss,
        'extra': extra,
        'timestamp': timestamp.toIso8601String(),
      };
}

// ─────────────────────────────────────────────────────────────────────────────
// TrainingMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Final metrics produced after training completes.
///
/// Classification metrics are populated for [ModelType.classifier].
/// Regression metrics are populated for [ModelType.regressor].
class TrainingMetrics {
  // ── Classification ──────────────────────────────────────────────────────
  final double? accuracy;
  final double? f1Score;        // macro-average
  final double? precision;      // macro-average
  final double? recall;         // macro-average
  final double? aucRoc;         // binary tasks only
  final List<List<int>>? confusionMatrix;

  // ── Regression ─────────────────────────────────────────────────────────
  final double? mse;
  final double? rmse;
  final double? mae;
  final double? rSquared;
  final double? mape;

  // ── Common ─────────────────────────────────────────────────────────────
  final double finalTrainLoss;
  final double finalValLoss;
  final int totalEpochs;
  final bool stoppedEarly;

  const TrainingMetrics({
    this.accuracy,
    this.f1Score,
    this.precision,
    this.recall,
    this.aucRoc,
    this.confusionMatrix,
    this.mse,
    this.rmse,
    this.mae,
    this.rSquared,
    this.mape,
    required this.finalTrainLoss,
    required this.finalValLoss,
    required this.totalEpochs,
    this.stoppedEarly = false,
  });

  factory TrainingMetrics.empty() => const TrainingMetrics(
        finalTrainLoss: 0,
        finalValLoss: 0,
        totalEpochs: 0,
      );

  // ── Pretty report ──────────────────────────────────────────────────────

  String toReport(ModelType modelType) {
    final b = StringBuffer();
    b.writeln('══════════════════════════════════════════');
    b.writeln('       offline_ml_pipeline — Results      ');
    b.writeln('══════════════════════════════════════════');
    b.writeln('Epochs     : $totalEpochs'
        '${stoppedEarly ? " (early stopped)" : ""}');
    b.writeln('Train Loss : ${finalTrainLoss.toStringAsFixed(6)}');
    b.writeln('Val   Loss : ${finalValLoss.toStringAsFixed(6)}');
    b.writeln('──────────────────────────────────────────');

    if (modelType == ModelType.classifier) {
      if (accuracy != null) {
        b.writeln('Accuracy   : ${(accuracy! * 100).toStringAsFixed(2)} %');
      }
      if (f1Score != null) {
        b.writeln('F1 Score   : ${f1Score!.toStringAsFixed(4)}');
      }
      if (precision != null) {
        b.writeln('Precision  : ${precision!.toStringAsFixed(4)}');
      }
      if (recall != null) {
        b.writeln('Recall     : ${recall!.toStringAsFixed(4)}');
      }
      if (aucRoc != null) {
        b.writeln('AUC-ROC    : ${aucRoc!.toStringAsFixed(4)}');
      }
      if (confusionMatrix != null) {
        b.writeln('Confusion Matrix:');
        for (final row in confusionMatrix!) {
          b.writeln('  ${row.map((v) => v.toString().padLeft(6)).join(" ")}');
        }
      }
    } else {
      if (rmse != null) b.writeln('RMSE       : ${rmse!.toStringAsFixed(4)}');
      if (mae != null)  b.writeln('MAE        : ${mae!.toStringAsFixed(4)}');
      if (rSquared != null) {
        b.writeln('R²         : ${rSquared!.toStringAsFixed(4)}');
      }
      if (mape != null) {
        b.writeln('MAPE       : ${(mape! * 100).toStringAsFixed(2)} %');
      }
    }

    b.writeln('══════════════════════════════════════════');
    return b.toString();
  }

  Map<String, dynamic> toJson() => {
        'accuracy': accuracy,
        'f1_score': f1Score,
        'precision': precision,
        'recall': recall,
        'auc_roc': aucRoc,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': rSquared,
        'mape': mape,
        'final_train_loss': finalTrainLoss,
        'final_val_loss': finalValLoss,
        'total_epochs': totalEpochs,
        'stopped_early': stoppedEarly,
      };
}

// ─────────────────────────────────────────────────────────────────────────────
// MetricsCalculator — post-training metric computation
// ─────────────────────────────────────────────────────────────────────────────

/// Computes all metrics from raw predictions and ground-truth labels.
class MetricsCalculator {
  MetricsCalculator._();

  // ── Classification ─────────────────────────────────────────────────────

  /// Computes accuracy, F1, precision, recall and confusion matrix.
  ///
  /// [preds]: predicted class indices, shape [N].
  /// [labels]: true class indices, shape [N].
  /// [numClasses]: total number of classes C.
  static ClassificationMetrics classification({
    required List<int> preds,
    required List<int> labels,
    required int numClasses,
  }) {
    assert(preds.length == labels.length);

    // Confusion matrix
    final cm = List.generate(
      numClasses,
      (_) => List<int>.filled(numClasses, 0),
    );
    for (int i = 0; i < preds.length; i++) {
      if (preds[i] < numClasses && labels[i] < numClasses) {
        cm[labels[i]][preds[i]]++;
      }
    }

    // Accuracy
    int correct = 0;
    for (int i = 0; i < preds.length; i++) {
      if (preds[i] == labels[i]) correct++;
    }
    final accuracy = correct / preds.length;

    // Per-class precision, recall, F1
    double sumP = 0, sumR = 0, sumF1 = 0;
    for (int c = 0; c < numClasses; c++) {
      int tp = cm[c][c];
      int fpCol = 0, fnRow = 0;
      for (int k = 0; k < numClasses; k++) {
        if (k != c) {
          fpCol += cm[k][c]; // false positive for class c
          fnRow += cm[c][k]; // false negative for class c
        }
      }
      final prec = (tp + fpCol) > 0 ? tp / (tp + fpCol) : 0.0;
      final rec = (tp + fnRow) > 0 ? tp / (tp + fnRow) : 0.0;
      final f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
      sumP += prec;
      sumR += rec;
      sumF1 += f1;
    }

    return ClassificationMetrics(
      accuracy: accuracy,
      precision: sumP / numClasses,
      recall: sumR / numClasses,
      f1Score: sumF1 / numClasses,
      confusionMatrix: cm,
    );
  }

  // ── Regression ─────────────────────────────────────────────────────────

  /// Computes MSE, RMSE, MAE, R², MAPE.
  static RegressionMetrics regression({
    required List<double> preds,
    required List<double> labels,
  }) {
    assert(preds.length == labels.length);
    final mse = LossFunctions.mse(preds, labels);
    return RegressionMetrics(
      mse: mse,
      rmse: math.sqrt(mse),
      mae: LossFunctions.mae(preds, labels),
      rSquared: LossFunctions.rSquared(preds, labels),
      mape: LossFunctions.mape(preds, labels),
    );
  }
}

/// Classification metric bundle returned by [MetricsCalculator.classification].
class ClassificationMetrics {
  final double accuracy;
  final double precision;
  final double recall;
  final double f1Score;
  final List<List<int>> confusionMatrix;

  const ClassificationMetrics({
    required this.accuracy,
    required this.precision,
    required this.recall,
    required this.f1Score,
    required this.confusionMatrix,
  });
}

/// Regression metric bundle returned by [MetricsCalculator.regression].
class RegressionMetrics {
  final double mse;
  final double rmse;
  final double mae;
  final double rSquared;
  final double mape;

  const RegressionMetrics({
    required this.mse,
    required this.rmse,
    required this.mae,
    required this.rSquared,
    required this.mape,
  });
}
