// ignore_for_file: lines_longer_than_80_chars

import 'dart:convert';
import 'dart:io';
import '../data/data_schema.dart';
import '../training/training_metrics.dart';

// ─────────────────────────────────────────────────────────────────────────────
// PipelineResult
// ─────────────────────────────────────────────────────────────────────────────

/// The complete outcome of a successful [MlPipeline.train] call.
///
/// Contains:
/// - Path to the exported `.tflite` (or `.json` for Dart-fallback) model.
/// - Full training metrics (accuracy, F1, RMSE, R², etc.).
/// - The [DataSchema] describing feature columns and class labels.
/// - Serialised normaliser + encoder state (needed for inference preprocessing).
/// - Training duration and per-epoch history.
class PipelineResult {
  /// Absolute path to the exported model file on the device.
  ///
  /// Extension is `.tflite` when ORT backend was used;
  /// `.json` for the Dart-fallback backend.
  final String modelPath;

  /// Final training and validation metrics.
  final TrainingMetrics metrics;

  /// Schema inferred from the CSV dataset.
  final DataSchema schema;

  /// Serialised [DataNormalizer] state — embed in inference pipeline so that
  /// raw user input can be normalised to match training distribution.
  final Map<String, dynamic> normalizerState;

  /// Serialised [FeatureEncoder] state — same as above for categorical encoding.
  final Map<String, dynamic> encoderState;

  /// Wall-clock duration of the training run.
  final Duration trainingDuration;

  /// Per-epoch loss history.
  final List<EpochRecord> epochHistory;

  /// Which backend produced this result.
  final String backend;

  const PipelineResult({
    required this.modelPath,
    required this.metrics,
    required this.schema,
    required this.normalizerState,
    required this.encoderState,
    required this.trainingDuration,
    required this.epochHistory,
    required this.backend,
  });

  // ── Convenience getters ─────────────────────────────────────────────────

  /// Convenience alias for [modelPath] (historical API compat).
  String get tflitePath => modelPath;

  /// Returns `true` if the model file exists on disk.
  bool get modelExists => File(modelPath).existsSync();

  /// Size of the model file in kilobytes.
  Future<double> get modelSizeKb async {
    final f = File(modelPath);
    if (!f.existsSync()) return 0.0;
    return (await f.length()) / 1024.0;
  }

  // ── Metadata persistence ────────────────────────────────────────────────

  /// Saves a JSON sidecar file next to [modelPath] containing all
  /// information needed to reconstruct the inference preprocessing pipeline.
  ///
  /// The sidecar path is `<model_base>_metadata.json`.
  Future<String> saveMetadata() async {
    final metaPath =
        modelPath.replaceAll(RegExp(r'\.(tflite|json|onnx)$'), '_metadata.json');

    final meta = {
      'offline_ml_pipeline_version': '0.1.0',
      'trained_at': DateTime.now().toIso8601String(),
      'backend': backend,
      'schema': schema.toJson(),
      'metrics': metrics.toJson(),
      'normalizerState': normalizerState,
      'encoderState': encoderState,
      'trainingDurationMs': trainingDuration.inMilliseconds,
      'epochCount': epochHistory.length,
      'modelPath': modelPath,
    };

    final file = File(metaPath);
    await file.create(recursive: true);
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(meta),
    );

    return metaPath;
  }

  // ── Loss curve export ───────────────────────────────────────────────────

  /// Exports a CSV file with `epoch,train_loss,val_loss` rows.
  Future<String> saveLossCurve() async {
    final csvPath =
        modelPath.replaceAll(RegExp(r'\.(tflite|json|onnx)$'), '_loss_curve.csv');

    final lines = StringBuffer('epoch,train_loss,val_loss\n');
    for (final r in epochHistory) {
      lines.writeln('${r.epoch},${r.trainLoss},${r.valLoss}');
    }

    await File(csvPath).writeAsString(lines.toString());
    return csvPath;
  }

  // ── Summary ─────────────────────────────────────────────────────────────

  @override
  String toString() {
    final buf = StringBuffer('PipelineResult {\n');
    buf.writeln('  modelPath  : $modelPath');
    buf.writeln('  backend    : $backend');
    buf.writeln('  epochs     : ${epochHistory.length}');
    buf.writeln('  duration   : ${trainingDuration.inSeconds}s');
    if (metrics.accuracy != null) {
      buf.writeln(
        '  accuracy   : ${(metrics.accuracy! * 100).toStringAsFixed(2)}%',
      );
    }
    if (metrics.rmse != null) {
      buf.writeln('  rmse       : ${metrics.rmse!.toStringAsFixed(4)}');
    }
    if (metrics.rSquared != null) {
      buf.writeln('  r²         : ${metrics.rSquared!.toStringAsFixed(4)}');
    }
    buf.write('}');
    return buf.toString();
  }
}
