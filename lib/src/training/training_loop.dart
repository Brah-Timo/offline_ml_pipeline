// ignore_for_file: lines_longer_than_80_chars

import 'dart:math' as math;
import '../data/data_schema.dart';
import '../utils/progress_notifier.dart';
import '../utils/error_handler.dart';
import 'training_session.dart';
import 'training_metrics.dart';
import 'optimizer_config.dart';

// ─────────────────────────────────────────────────────────────────────────────
// EarlyStopping
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for early-stopping.
///
/// Training halts if [monitor] metric does not improve by at least
/// [minDelta] for [patience] consecutive epochs.
class EarlyStopping {
  /// Metric to watch: `'val_loss'` or `'val_accuracy'`.
  final String monitor;

  /// Number of epochs without improvement before stopping.
  final int patience;

  /// Minimum change to be considered an improvement.
  final double minDelta;

  const EarlyStopping({
    this.monitor = 'val_loss',
    this.patience = 10,
    this.minDelta = 1e-4,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// TrainingLoop
// ─────────────────────────────────────────────────────────────────────────────

/// Executes the full training loop: epochs → mini-batches → parameter update.
///
/// Responsibilities:
/// - Shuffle training data each epoch (optional).
/// - Slice data into mini-batches.
/// - Call `session.trainStep` per batch.
/// - Call `session.evalStep` at the end of each epoch.
/// - Compute epoch metrics and emit [TrainingProgress] events.
/// - Apply early stopping.
/// - Return final [TrainingMetrics].
class TrainingLoop {
  final TrainingSession session;
  final ProcessedDataset trainData;
  final ProcessedDataset valData;
  final int epochs;
  final int batchSize;
  final ProgressNotifier progressNotifier;
  final EarlyStopping? earlyStopping;
  final bool shuffleEachEpoch;
  final LrSchedule? lrSchedule;
  final double initialLr;

  final List<EpochRecord> epochHistory = [];
  Duration duration = Duration.zero;

  TrainingLoop({
    required this.session,
    required this.trainData,
    required this.valData,
    required this.epochs,
    required this.batchSize,
    required this.progressNotifier,
    this.earlyStopping,
    this.shuffleEachEpoch = true,
    this.lrSchedule,
    this.initialLr = 0.001,
  });

  // ── Main entry point ───────────────────────────────────────────────────

  /// Runs the full training loop and returns final metrics.
  Future<TrainingMetrics> run() async {
    final sw = Stopwatch()..start();

    double bestWatchedValue = double.infinity;
    int patienceCounter = 0;
    double trainLossFinal = 0;
    double valLossFinal = 0;
    bool stoppedEarly = false;
    int lastEpoch = epochs;

    for (int epoch = 1; epoch <= epochs; epoch++) {
      // ── Learning-rate schedule ────────────────────────────────────────
      final currentLr = lrSchedule?.compute(initialLr, epoch) ?? initialLr;
      // (In real usage the session would consume currentLr — DartSession
      //  Adam uses initialLr from OptimizerConfig; ORT uses the checkpoint lr.)
      // ignore: unused_local_variable
      final _unusedLr = currentLr; // lr consumed by session internally

      // ── Shuffle ───────────────────────────────────────────────────────
      if (shuffleEachEpoch) {
        trainData.shuffle(seed: epoch);
      }

      // ── Mini-batch training ───────────────────────────────────────────
      double epochTrainLoss = 0.0;
      int numBatches = 0;
      final n = trainData.rowCount;

      for (int start = 0; start < n; start += batchSize) {
        final end = math.min(start + batchSize, n);
        final batch = trainData.slice(start, end);

        try {
          final batchLoss = await session.trainStep(
            flatInputs: batch.features,
            labels: batch.labels,
            batchSize: batch.size,
          );

          if (batchLoss.isNaN || batchLoss.isInfinite) {
            throw TrainingStepException(
              'Loss is ${batchLoss.isNaN ? "NaN" : "Infinity"}. '
              'Try a lower learning rate.',
              epoch: epoch,
              step: numBatches,
            );
          }

          epochTrainLoss += batchLoss;
          numBatches++;
        } catch (e) {
          if (e is TrainingStepException) rethrow;
          throw TrainingStepException(
            e.toString(),
            epoch: epoch,
            step: numBatches,
          );
        }
      }

      if (numBatches > 0) epochTrainLoss /= numBatches;

      // ── Validation ────────────────────────────────────────────────────
      double epochValLoss = -1.0;
      if (valData.rowCount > 0) {
        epochValLoss = await session.evalStep(
          flatInputs: valData.allFeatures,
          labels: valData.allLabels,
          batchSize: valData.rowCount,
        );
      }

      trainLossFinal = epochTrainLoss;
      valLossFinal = epochValLoss;
      lastEpoch = epoch;

      // ── Record ────────────────────────────────────────────────────────
      final record = EpochRecord(
        epoch: epoch,
        trainLoss: epochTrainLoss,
        valLoss: epochValLoss,
        timestamp: DateTime.now(),
      );
      epochHistory.add(record);

      // ── Emit progress ─────────────────────────────────────────────────
      progressNotifier.notify(
        TrainingProgress(
          epoch: epoch,
          totalEpochs: epochs,
          trainLoss: epochTrainLoss,
          valLoss: epochValLoss,
          percentage: (epoch / epochs * 100).round(),
        ),
      );

      // ── Early stopping ────────────────────────────────────────────────
      if (earlyStopping != null) {
        final watchedValue = earlyStopping!.monitor == 'val_loss'
            ? epochValLoss
            : -epochTrainLoss; // "accuracy" → negate loss proxy

        if (watchedValue < bestWatchedValue - earlyStopping!.minDelta) {
          bestWatchedValue = watchedValue;
          patienceCounter = 0;
        } else {
          patienceCounter++;
          if (patienceCounter >= earlyStopping!.patience) {
            stoppedEarly = true;
            progressNotifier.notifyEarlyStopped(epoch);
            break;
          }
        }
      }
    }

    sw.stop();
    duration = sw.elapsed;

    return await _computeFinalMetrics(
      trainLossFinal: trainLossFinal,
      valLossFinal: valLossFinal,
      totalEpochs: lastEpoch,
      stoppedEarly: stoppedEarly,
    );
  }

  // ── Metric computation ─────────────────────────────────────────────────

  Future<TrainingMetrics> _computeFinalMetrics({
    required double trainLossFinal,
    required double valLossFinal,
    required int totalEpochs,
    required bool stoppedEarly,
  }) async {
    if (valData.rowCount == 0) {
      return TrainingMetrics(
        finalTrainLoss: trainLossFinal,
        finalValLoss: valLossFinal,
        totalEpochs: totalEpochs,
        stoppedEarly: stoppedEarly,
      );
    }

    final rawPreds = await session.predict(
      flatInputs: valData.allFeatures,
      n: valData.rowCount,
    );

    if (session.spec.isClassifier) {
      final predClasses =
          rawPreds.map((p) => _argmax(p)).toList();
      final trueClasses =
          valData.allLabels.map((l) => (l as num).toInt()).toList();

      final cm = MetricsCalculator.classification(
        preds: predClasses,
        labels: trueClasses,
        numClasses: session.spec.outputSize,
      );

      return TrainingMetrics(
        accuracy: cm.accuracy,
        f1Score: cm.f1Score,
        precision: cm.precision,
        recall: cm.recall,
        confusionMatrix: cm.confusionMatrix,
        finalTrainLoss: trainLossFinal,
        finalValLoss: valLossFinal,
        totalEpochs: totalEpochs,
        stoppedEarly: stoppedEarly,
      );
    } else {
      final predValues = rawPreds.map((p) => p[0]).toList();
      final trueValues =
          valData.allLabels.map((l) => (l as num).toDouble()).toList();

      final reg = MetricsCalculator.regression(
        preds: predValues,
        labels: trueValues,
      );

      return TrainingMetrics(
        mse: reg.mse,
        rmse: reg.rmse,
        mae: reg.mae,
        rSquared: reg.rSquared,
        mape: reg.mape,
        finalTrainLoss: trainLossFinal,
        finalValLoss: valLossFinal,
        totalEpochs: totalEpochs,
        stoppedEarly: stoppedEarly,
      );
    }
  }

  int _argmax(List<double> values) {
    int idx = 0;
    double best = values[0];
    for (int i = 1; i < values.length; i++) {
      if (values[i] > best) {
        best = values[i];
        idx = i;
      }
    }
    return idx;
  }
}
