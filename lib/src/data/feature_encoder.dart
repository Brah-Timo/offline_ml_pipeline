// ignore_for_file: lines_longer_than_80_chars

import 'csv_loader.dart';
import 'data_schema.dart';

// ─────────────────────────────────────────────────────────────────────────────
// FeatureEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Encodes categorical (string) columns in a [RawDataTable] into numeric
/// values so the model can consume them.
///
/// Two encoding modes are supported per column:
/// - **Ordinal** (default): each unique category → consecutive integer (0, 1, 2 …).
/// - **One-Hot**: each category → a new binary column (existing column removed).
///
/// The encoder is *fit* once on training data, then *applied* consistently
/// to validation / test data, preventing unseen-category surprises.
///
/// Example:
/// ```dart
/// final enc    = FeatureEncoder(schema: schema);
/// final train2 = enc.fitTransform(trainTable);
/// final val2   = enc.transform(valTable);
/// ```
class FeatureEncoder {
  final DataSchema schema;

  /// Maps column name → sorted list of unique categories discovered at fit time.
  final Map<String, List<String>> _vocabulary = {};

  /// Which encoding mode to use per categorical column.
  final Map<String, _EncodingMode> _encodingModes = {};

  /// After encoding, contains the *expanded* column order
  /// (accounts for one-hot expansion).
  late List<String> _encodedHeaders;

  bool _fitted = false;

  FeatureEncoder({
    required this.schema,
    EncodingStrategy defaultStrategy = EncodingStrategy.ordinal,
  }) {
    for (final col in schema.featureColumns) {
      if (schema.isCategorical(col)) {
        _encodingModes[col] = defaultStrategy == EncodingStrategy.ordinal
            ? _EncodingMode.ordinal
            : _EncodingMode.oneHot;
      }
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────

  /// Fits vocabulary on [table] and returns the encoded table.
  RawDataTable fitTransform(RawDataTable table) {
    _fit(table);
    return _applyTransform(table);
  }

  /// Applies vocabulary fitted on training data to a new table.
  RawDataTable transform(RawDataTable table) {
    if (!_fitted) {
      throw StateError(
          'FeatureEncoder has not been fitted. Call fitTransform() first.');
    }
    return _applyTransform(table);
  }

  /// Returns the list of column headers after encoding.
  List<String> get encodedHeaders {
    if (!_fitted) return schema.headers;
    return _encodedHeaders;
  }

  /// Serialises vocabulary to a JSON-compatible map for persistence.
  Map<String, dynamic> serialize() => {
        'vocabulary': _vocabulary,
        'encodingModes': _encodingModes.map(
          (k, v) => MapEntry(k, v.name),
        ),
        'encodedHeaders': _encodedHeaders,
      };

  /// Restores an encoder from serialised state.
  factory FeatureEncoder.fromJson(
    Map<String, dynamic> json,
    DataSchema schema,
  ) {
    final enc = FeatureEncoder(schema: schema);
    (json['vocabulary'] as Map<String, dynamic>).forEach((col, raw) {
      enc._vocabulary[col] = List<String>.from(raw as List);
    });
    (json['encodingModes'] as Map<String, dynamic>).forEach((col, raw) {
      enc._encodingModes[col] = _EncodingMode.values.firstWhere(
        (e) => e.name == raw,
      );
    });
    enc._encodedHeaders =
        List<String>.from(json['encodedHeaders'] as List);
    enc._fitted = true;
    return enc;
  }

  // ── Internal helpers ───────────────────────────────────────────────────

  void _fit(RawDataTable table) {
    for (final col in schema.featureColumns) {
      if (!schema.isCategorical(col)) continue;

      final colIdx = table.headers.indexOf(col);
      final unique = <String>{};
      for (final row in table.rows) {
        unique.add(row[colIdx].toString().trim());
      }
      _vocabulary[col] = unique.toList()..sort();
    }

    // Build encoded header list
    final headers = <String>[];
    for (final h in table.headers) {
      if (_vocabulary.containsKey(h) &&
          _encodingModes[h] == _EncodingMode.oneHot) {
        // One-hot expansion: replace original column with N binary columns
        for (final cat in _vocabulary[h]!) {
          headers.add('${h}_$cat');
        }
      } else {
        headers.add(h);
      }
    }
    _encodedHeaders = headers;
    _fitted = true;
  }

  RawDataTable _applyTransform(RawDataTable table) {
    final newRows = <List<dynamic>>[];

    for (final row in table.rows) {
      final newRow = <dynamic>[];

      for (int ci = 0; ci < table.headers.length; ci++) {
        final col = table.headers[ci];

        if (!_vocabulary.containsKey(col)) {
          // Not a categorical column → pass through
          newRow.add(row[ci]);
          continue;
        }

        final value = row[ci].toString().trim();
        final vocab = _vocabulary[col]!;
        final mode = _encodingModes[col]!;

        if (mode == _EncodingMode.ordinal) {
          // Map to integer index; unknown category → last index + 1
          final idx = vocab.indexOf(value);
          newRow.add(idx == -1 ? vocab.length : idx);
        } else {
          // One-hot: emit one 0/1 per category
          for (final cat in vocab) {
            newRow.add(value == cat ? 1 : 0);
          }
        }
      }

      newRows.add(newRow);
    }

    return RawDataTable(
      headers: _encodedHeaders,
      rows: newRows,
      sourcePath: table.sourcePath,
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Enumerations
// ─────────────────────────────────────────────────────────────────────────────

/// High-level strategy choice exposed to the user.
enum EncodingStrategy {
  /// Each category maps to a single integer (0, 1, 2 …).
  ordinal,

  /// Each category expands into its own binary column.
  oneHot,
}

enum _EncodingMode { ordinal, oneHot }
