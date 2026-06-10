// ignore_for_file: lines_longer_than_80_chars

import '../utils/error_handler.dart';

// ─────────────────────────────────────────────────────────────────────────────
// ColumnType
// ─────────────────────────────────────────────────────────────────────────────

/// Fundamental type assigned to each column in a dataset.
enum ColumnType {
  /// Column values are (or can be cast to) double.
  numeric,

  /// Column values are discrete string categories (≤ 30 unique values).
  categorical,

  /// Free-form text — excluded from feature extraction automatically.
  text,
}

// ─────────────────────────────────────────────────────────────────────────────
// DataSchema
// ─────────────────────────────────────────────────────────────────────────────

/// Describes the structure of a dataset: column names, types,
/// target column metadata, and class label strings.
///
/// A [DataSchema] is produced automatically by [RawDataTable.inferSchema].
/// Advanced users may also construct one manually.
class DataSchema {
  /// All column names in the original CSV order.
  final List<String> headers;

  /// Inferred type for every column.
  final Map<String, ColumnType> columnTypes;

  /// Name of the column the model should predict.
  final String targetColumn;

  /// Zero-based index of [targetColumn] in [headers].
  final int targetIndex;

  /// Number of distinct output classes (for classifiers) or 1 (for regressors).
  final int outputClasses;

  /// Sorted list of unique string labels for the target column.
  /// For regression targets this list is empty.
  final List<String> classLabels;

  const DataSchema({
    required this.headers,
    required this.columnTypes,
    required this.targetColumn,
    required this.targetIndex,
    required this.outputClasses,
    required this.classLabels,
  });

  // ── Derived helpers ────────────────────────────────────────────────────

  /// Column names that will be used as model input features
  /// (everything except [targetColumn] and [ColumnType.text] columns).
  List<String> get featureColumns => headers
      .where(
        (h) =>
            h != targetColumn &&
            columnTypes[h] != ColumnType.text,
      )
      .toList();

  /// Count of input features available to the model.
  int get featureCount => featureColumns.length;

  /// Returns `true` if [column] is categorical.
  bool isCategorical(String column) =>
      columnTypes[column] == ColumnType.categorical;

  /// Returns `true` if [column] is numeric.
  bool isNumeric(String column) =>
      columnTypes[column] == ColumnType.numeric;

  /// Converts a string class label to its integer index.
  ///
  /// Throws [UnknownLabelException] if the label is not in [classLabels].
  int labelToIndex(String label) {
    final idx = classLabels.indexOf(label);
    if (idx == -1) throw UnknownLabelException(label, classLabels);
    return idx;
  }

  /// Converts an integer class index back to its string label.
  String indexToLabel(int index) {
    if (index < 0 || index >= classLabels.length) {
      throw RangeError.index(index, classLabels, 'classLabels');
    }
    return classLabels[index];
  }

  // ── Serialisation ──────────────────────────────────────────────────────

  Map<String, dynamic> toJson() => {
        'headers': headers,
        'columnTypes': columnTypes.map(
          (k, v) => MapEntry(k, v.name),
        ),
        'targetColumn': targetColumn,
        'targetIndex': targetIndex,
        'outputClasses': outputClasses,
        'classLabels': classLabels,
      };

  factory DataSchema.fromJson(Map<String, dynamic> json) {
    return DataSchema(
      headers: List<String>.from(json['headers'] as List),
      columnTypes: (json['columnTypes'] as Map<String, dynamic>).map(
        (k, v) => MapEntry(
          k,
          ColumnType.values.firstWhere((e) => e.name == v),
        ),
      ),
      targetColumn: json['targetColumn'] as String,
      targetIndex: json['targetIndex'] as int,
      outputClasses: json['outputClasses'] as int,
      classLabels: List<String>.from(json['classLabels'] as List),
    );
  }

  @override
  String toString() => 'DataSchema('
      'features: $featureCount, '
      'target: "$targetColumn", '
      'classes: $outputClasses)';
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessedDataset  (used by training layer)
// ─────────────────────────────────────────────────────────────────────────────

/// Holds a split portion of the dataset as flat float lists,
/// ready to be handed to the training session.
class ProcessedDataset {
  /// Feature values laid out as [rowCount × featureCount] flat list.
  final List<double> featuresFlat;

  /// Integer class indices (classifier) or raw doubles (regressor).
  final List<dynamic> labels;

  final int rowCount;
  final int featureCount;

  ProcessedDataset({
    required this.featuresFlat,
    required this.labels,
    required this.rowCount,
    required this.featureCount,
  }) : assert(featuresFlat.length == rowCount * featureCount);

  /// Returns the feature vector for row [i].
  List<double> rowFeatures(int i) {
    final start = i * featureCount;
    return featuresFlat.sublist(start, start + featureCount);
  }

  /// Flat feature list for the entire dataset.
  List<double> get allFeatures => featuresFlat;

  /// All labels.
  List<dynamic> get allLabels => labels;

  /// Extracts a contiguous slice [from, to) as a [DataBatch].
  DataBatch slice(int from, int to) {
    final batchRows = to - from;
    return DataBatch(
      features: featuresFlat.sublist(from * featureCount, to * featureCount),
      labels: labels.sublist(from, to),
      size: batchRows,
      featureCount: featureCount,
    );
  }

  /// In-place Fisher-Yates shuffle (keeps features and labels aligned).
  void shuffle({int? seed}) {
    final rng = seed != null
        ? _SeededRandom(seed)
        : _SeededRandom(DateTime.now().millisecondsSinceEpoch);

    for (int i = rowCount - 1; i > 0; i--) {
      final j = rng.nextInt(i + 1);
      _swapRows(i, j);
    }
  }

  void _swapRows(int a, int b) {
    // Swap labels
    final tmpLabel = labels[a];
    labels[a] = labels[b];
    labels[b] = tmpLabel;
    // Swap feature block
    for (int k = 0; k < featureCount; k++) {
      final tmpF = featuresFlat[a * featureCount + k];
      featuresFlat[a * featureCount + k] =
          featuresFlat[b * featureCount + k];
      featuresFlat[b * featureCount + k] = tmpF;
    }
  }
}

/// A mini-batch extracted from a [ProcessedDataset].
class DataBatch {
  final List<double> features; // flat, size × featureCount
  final List<dynamic> labels;
  final int size;
  final int featureCount;

  const DataBatch({
    required this.features,
    required this.labels,
    required this.size,
    required this.featureCount,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal seeded RNG (avoids dart:math import cycle)
// ─────────────────────────────────────────────────────────────────────────────
class _SeededRandom {
  int _state;

  _SeededRandom(this._state);

  int nextInt(int max) {
    _state = (_state * 1664525 + 1013904223) & 0xFFFFFFFF;
    return _state % max;
  }
}
