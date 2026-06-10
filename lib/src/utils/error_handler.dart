// ─────────────────────────────────────────────────────────────────────────────
// Base exception
// ─────────────────────────────────────────────────────────────────────────────

/// Root exception type for all errors thrown by offline_ml_pipeline.
///
/// Catch this to handle all package errors generically, or catch specific
/// subtypes for fine-grained error handling.
class OfflineMlException implements Exception {
  final String message;
  final Object? cause;
  final StackTrace? causeStackTrace;

  const OfflineMlException(
    this.message, {
    this.cause,
    this.causeStackTrace,
  });

  @override
  String toString() {
    final buf = StringBuffer('OfflineMlException: $message');
    if (cause != null) buf.write('\n  Caused by: $cause');
    return buf.toString();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data errors
// ─────────────────────────────────────────────────────────────────────────────

/// Column name requested by the user does not exist in the dataset.
class ColumnNotFoundException extends OfflineMlException {
  final String column;
  final List<String> availableColumns;

  ColumnNotFoundException(this.column, this.availableColumns)
      : super(
          'Column "$column" not found. '
          'Available columns: ${availableColumns.join(", ")}',
        );
}

/// A class label was encountered that was not seen during training.
class UnknownLabelException extends OfflineMlException {
  final String label;
  final List<String> knownLabels;

  UnknownLabelException(this.label, this.knownLabels)
      : super(
          'Unknown label "$label". '
          'Known labels: ${knownLabels.join(", ")}',
        );
}

// ─────────────────────────────────────────────────────────────────────────────
// ORT / native errors
// ─────────────────────────────────────────────────────────────────────────────

/// Thrown when ONNX Runtime returns a non-OK status.
class OrtException extends OfflineMlException {
  final int? errorCode;

  OrtException(String message, {this.errorCode, Object? cause})
      : super(message, cause: cause);

  @override
  String toString() {
    final code = errorCode != null ? ' [code: $errorCode]' : '';
    return 'OrtException$code: $message';
  }
}

/// Thrown when the current platform is not supported by the package.
class UnsupportedPlatformException extends OfflineMlException {
  final String platform;

  UnsupportedPlatformException(this.platform)
      : super(
          'Platform "$platform" is not supported by offline_ml_pipeline. '
          'Supported platforms: android, ios, macos, windows, linux.',
        );
}

// ─────────────────────────────────────────────────────────────────────────────
// Training errors
// ─────────────────────────────────────────────────────────────────────────────

/// Thrown when training artifacts cannot be loaded or built.
class ArtifactLoadException extends OfflineMlException {
  final String artifactPath;

  ArtifactLoadException(this.artifactPath, {Object? cause})
      : super(
          'Failed to load training artifact: "$artifactPath"',
          cause: cause,
        );
}

/// Thrown when a training step fails (NaN loss, OOM, etc.).
class TrainingStepException extends OfflineMlException {
  final int epoch;
  final int step;

  TrainingStepException(String message, {required this.epoch, required this.step})
      : super('Training failed at epoch $epoch, step $step: $message');
}

// ─────────────────────────────────────────────────────────────────────────────
// Export errors
// ─────────────────────────────────────────────────────────────────────────────

/// Thrown when the model export (ONNX → TFLite) fails.
class ModelExportException extends OfflineMlException {
  final String outputPath;

  ModelExportException(String message, {required this.outputPath, Object? cause})
      : super('Model export failed → "$outputPath": $message', cause: cause);
}
