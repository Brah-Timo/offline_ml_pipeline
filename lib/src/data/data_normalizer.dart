// ignore_for_file: lines_longer_than_80_chars

import 'dart:math' as math;
import 'csv_loader.dart';
import 'data_schema.dart';

// ─────────────────────────────────────────────────────────────────────────────
// NormalizationStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Normalisation strategy applied to numeric columns.
enum NormalizationStrategy {
  /// Scale each feature to [0, 1].
  ///
  /// Formula: `(x − min) / (max − min)`
  minMax,

  /// Standardise to zero mean, unit variance.
  ///
  /// Formula: `(x − μ) / σ`
  zScore,

  /// No normalisation — pass values through unchanged.
  none,
}

// ─────────────────────────────────────────────────────────────────────────────
// ColumnNormParams
// ─────────────────────────────────────────────────────────────────────────────

/// Per-column statistics captured during `fit()`.
class ColumnNormParams {
  final double min;
  final double max;
  final double mean;
  final double std;

  const ColumnNormParams({
    required this.min,
    required this.max,
    required this.mean,
    required this.std,
  });

  Map<String, dynamic> toJson() => {
        'min': min,
        'max': max,
        'mean': mean,
        'std': std,
      };

  factory ColumnNormParams.fromJson(Map<String, dynamic> j) =>
      ColumnNormParams(
        min: (j['min'] as num).toDouble(),
        max: (j['max'] as num).toDouble(),
        mean: (j['mean'] as num).toDouble(),
        std: (j['std'] as num).toDouble(),
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// DataNormalizer
// ─────────────────────────────────────────────────────────────────────────────

/// Fits normalisation parameters on a training set and applies the same
/// transform to validation / test sets (no data leakage).
///
/// Usage:
/// ```dart
/// final norm = DataNormalizer(strategy: NormalizationStrategy.zScore);
/// final train = norm.fitTransform(trainTable, schema);
/// final val   = norm.transform(valTable, schema);
/// ```
class DataNormalizer {
  final NormalizationStrategy strategy;

  /// Per-column fit parameters; populated after [fit] or [fitTransform].
  final Map<String, ColumnNormParams> _params = {};

  bool _fitted = false;

  DataNormalizer({this.strategy = NormalizationStrategy.minMax});

  // ── Public API ─────────────────────────────────────────────────────────

  /// Fits statistics on [table] and returns a new table with
  /// numeric columns normalised.
  RawDataTable fitTransform(RawDataTable table, DataSchema schema) {
    _fit(table, schema);
    return _applyTransform(table, schema);
  }

  /// Applies the previously fitted parameters to a new table.
  ///
  /// Must call [fit] or [fitTransform] first.
  RawDataTable transform(RawDataTable table, DataSchema schema) {
    if (!_fitted) {
      throw StateError('DataNormalizer has not been fitted yet. '
          'Call fitTransform() on the training data first.');
    }
    return _applyTransform(table, schema);
  }

  /// Normalises a single raw value using the parameters for [column].
  double normalizeValue(double raw, String column) {
    if (!_fitted || !_params.containsKey(column)) return raw;
    return _applyStrategy(raw, _params[column]!);
  }

  /// Reverses the normalisation for a single value.
  double denormalizeValue(double scaled, String column) {
    if (!_fitted || !_params.containsKey(column)) return scaled;
    final p = _params[column]!;
    switch (strategy) {
      case NormalizationStrategy.minMax:
        return scaled * (p.max - p.min) + p.min;
      case NormalizationStrategy.zScore:
        return scaled * p.std + p.mean;
      case NormalizationStrategy.none:
        return scaled;
    }
  }

  /// Serialises fit parameters to a JSON-compatible map.
  Map<String, dynamic> serialize() => {
        'strategy': strategy.name,
        'params': _params.map(
          (k, v) => MapEntry(k, v.toJson()),
        ),
      };

  /// Restores a [DataNormalizer] from serialised state.
  factory DataNormalizer.fromJson(Map<String, dynamic> json) {
    final strategy = NormalizationStrategy.values.firstWhere(
      (e) => e.name == json['strategy'],
    );
    final norm = DataNormalizer(strategy: strategy);
    (json['params'] as Map<String, dynamic>).forEach((col, raw) {
      norm._params[col] =
          ColumnNormParams.fromJson(raw as Map<String, dynamic>);
    });
    norm._fitted = true;
    return norm;
  }

  // ── Internal helpers ───────────────────────────────────────────────────

  void _fit(RawDataTable table, DataSchema schema) {
    if (strategy == NormalizationStrategy.none) {
      _fitted = true;
      return;
    }

    for (final col in schema.featureColumns) {
      if (!schema.isNumeric(col)) continue;

      final values = <double>[];
      final colIdx = table.headers.indexOf(col);
      for (final row in table.rows) {
        final v = double.tryParse(row[colIdx].toString());
        if (v != null && v.isFinite) values.add(v);
      }

      if (values.isEmpty) continue;

      final min = values.reduce(math.min);
      final max = values.reduce(math.max);
      final mean = values.reduce((a, b) => a + b) / values.length;
      final variance = values
              .map((v) => math.pow(v - mean, 2).toDouble())
              .reduce((a, b) => a + b) /
          values.length;
      final std = math.sqrt(variance).clamp(1e-9, double.infinity);

      _params[col] = ColumnNormParams(
        min: min,
        max: max,
        mean: mean,
        std: std,
      );
    }

    _fitted = true;
  }

  RawDataTable _applyTransform(RawDataTable table, DataSchema schema) {
    if (strategy == NormalizationStrategy.none) return table;

    final newRows = table.rows.map((row) {
      final newRow = List<dynamic>.from(row);
      for (final col in schema.featureColumns) {
        if (!schema.isNumeric(col)) continue;
        if (!_params.containsKey(col)) continue;

        final colIdx = table.headers.indexOf(col);
        final raw = double.tryParse(newRow[colIdx].toString()) ?? 0.0;
        newRow[colIdx] = _applyStrategy(raw, _params[col]!);
      }
      return newRow;
    }).toList();

    return RawDataTable(
      headers: table.headers,
      rows: newRows,
      sourcePath: table.sourcePath,
    );
  }

  double _applyStrategy(double raw, ColumnNormParams p) {
    switch (strategy) {
      case NormalizationStrategy.minMax:
        final range = p.max - p.min;
        if (range < 1e-9) return 0.0;
        return ((raw - p.min) / range).clamp(0.0, 1.0);
      case NormalizationStrategy.zScore:
        return (raw - p.mean) / p.std;
      case NormalizationStrategy.none:
        return raw;
    }
  }
}
