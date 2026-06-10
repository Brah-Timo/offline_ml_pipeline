import 'dart:async';

// ─────────────────────────────────────────────────────────────────────────────
// TrainingProgress
// ─────────────────────────────────────────────────────────────────────────────

/// Emitted by [ProgressNotifier] after every training epoch.
class TrainingProgress {
  /// Current epoch number (1-based).
  final int epoch;

  /// Total number of epochs configured.
  final int totalEpochs;

  /// Average training loss for this epoch.
  final double trainLoss;

  /// Validation loss for this epoch (−1 if not evaluated).
  final double valLoss;

  /// Completion percentage (0–100).
  final int percentage;

  /// Optional per-epoch metrics (accuracy, f1, etc.).
  final Map<String, double> extraMetrics;

  const TrainingProgress({
    required this.epoch,
    required this.totalEpochs,
    required this.trainLoss,
    this.valLoss = -1.0,
    required this.percentage,
    this.extraMetrics = const {},
  });

  @override
  String toString() =>
      'Epoch $epoch/$totalEpochs | '
      'train_loss=${trainLoss.toStringAsFixed(4)} | '
      'val_loss=${valLoss < 0 ? "N/A" : valLoss.toStringAsFixed(4)} | '
      '$percentage%';
}

// ─────────────────────────────────────────────────────────────────────────────
// ProgressNotifier
// ─────────────────────────────────────────────────────────────────────────────

/// Internal broadcast stream controller that carries [TrainingProgress]
/// events from the training loop back to the consumer.
///
/// The [stream] is exposed on [MlPipeline.progressStream].
class ProgressNotifier {
  final _controller = StreamController<TrainingProgress>.broadcast();

  /// The stream that consumers listen to.
  Stream<TrainingProgress> get stream => _controller.stream;

  /// Emits a progress update.
  void notify(TrainingProgress progress) {
    if (!_controller.isClosed) _controller.add(progress);
  }

  /// Emits a final update tagged as early-stopped.
  void notifyEarlyStopped(int stoppedEpoch) {
    if (!_controller.isClosed) {
      _controller.add(
        TrainingProgress(
          epoch: stoppedEpoch,
          totalEpochs: stoppedEpoch,
          trainLoss: 0.0,
          percentage: 100,
          extraMetrics: const {'early_stopped': 1.0},
        ),
      );
    }
  }

  /// Closes the underlying stream. Called when the pipeline is disposed.
  void close() => _controller.close();
}
