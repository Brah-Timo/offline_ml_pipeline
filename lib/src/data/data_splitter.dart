// ignore_for_file: lines_longer_than_80_chars

import 'dart:math' as math;
import 'csv_loader.dart';
import 'data_schema.dart';

// ─────────────────────────────────────────────────────────────────────────────
// DataSplits
// ─────────────────────────────────────────────────────────────────────────────

/// Container for the three dataset splits produced by [DataSplitter].
class DataSplits {
  final ProcessedDataset train;
  final ProcessedDataset validation;
  final ProcessedDataset test;

  const DataSplits({
    required this.train,
    required this.validation,
    required this.test,
  });

  @override
  String toString() =>
      'DataSplits(train: ${train.rowCount}, '
      'val: ${validation.rowCount}, '
      'test: ${test.rowCount})';
}

// ─────────────────────────────────────────────────────────────────────────────
// DataSplitter
// ─────────────────────────────────────────────────────────────────────────────

/// Converts a normalised [RawDataTable] into train / validation / test
/// [ProcessedDataset] splits.
///
/// Supports optional shuffling with a reproducible seed.
///
/// Ratios must sum to 1.0.
///
/// Example:
/// ```dart
/// final splitter = DataSplitter(trainRatio: 0.7, valRatio: 0.15, testRatio: 0.15);
/// final splits   = splitter.split(normalizedTable, schema);
/// ```
class DataSplitter {
  final double trainRatio;
  final double valRatio;
  final double testRatio;
  final bool shuffle;
  final int? seed;

  DataSplitter({
    this.trainRatio = 0.8,
    this.valRatio = 0.1,
    this.testRatio = 0.1,
    this.shuffle = true,
    this.seed,
  }) : assert(
          (trainRatio + valRatio + testRatio - 1.0).abs() < 1e-6,
          'trainRatio + valRatio + testRatio must equal 1.0',
        );

  /// Splits [table] according to the configured ratios.
  DataSplits split(RawDataTable table, DataSchema schema) {
    final indices = List<int>.generate(table.rowCount, (i) => i);

    if (shuffle) {
      final rng = math.Random(seed ?? DateTime.now().millisecondsSinceEpoch);
      for (int i = indices.length - 1; i > 0; i--) {
        final j = rng.nextInt(i + 1);
        final tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }
    }

    final n = indices.length;
    final nTrain = (n * trainRatio).round();
    final nVal = (n * valRatio).round();

    final trainIdx = indices.sublist(0, nTrain);
    final valIdx = indices.sublist(nTrain, nTrain + nVal);
    final testIdx = indices.sublist(nTrain + nVal);

    return DataSplits(
      train: _buildDataset(table, schema, trainIdx),
      validation: _buildDataset(table, schema, valIdx),
      test: _buildDataset(table, schema, testIdx),
    );
  }

  // ── Internal helpers ───────────────────────────────────────────────────

  ProcessedDataset _buildDataset(
    RawDataTable table,
    DataSchema schema,
    List<int> indices,
  ) {
    final featureCols = schema.featureColumns;
    final featureCount = featureCols.length;
    final colIndices = featureCols.map((c) => table.headers.indexOf(c)).toList();

    final featuresFlat = <double>[];
    final labels = <dynamic>[];

    for (final i in indices) {
      final row = table.rows[i];

      // Features
      for (final ci in colIndices) {
        final raw = row[ci];
        if (schema.isNumeric(featureCols[colIndices.indexOf(ci)])) {
          featuresFlat.add(double.tryParse(raw.toString()) ?? 0.0);
        } else {
          // Categorical encoded columns arrive as numeric after FeatureEncoder
          featuresFlat.add(double.tryParse(raw.toString()) ?? 0.0);
        }
      }

      // Label
      final rawLabel = row[schema.targetIndex].toString().trim();
      if (schema.classLabels.isNotEmpty) {
        // Classifier: convert label string → integer index
        labels.add(schema.labelToIndex(rawLabel));
      } else {
        // Regressor: keep as double
        labels.add(double.tryParse(rawLabel) ?? 0.0);
      }
    }

    return ProcessedDataset(
      featuresFlat: featuresFlat,
      labels: labels,
      rowCount: indices.length,
      featureCount: featureCount,
    );
  }
}
