// ignore_for_file: lines_longer_than_80_chars

import 'dart:io';
import 'package:csv/csv.dart';
import 'data_schema.dart';
import '../utils/error_handler.dart';

// ─────────────────────────────────────────────────────────────────────────────
// CsvLoader
// ─────────────────────────────────────────────────────────────────────────────

/// Reads a CSV file from the device and returns a [RawDataTable].
///
/// Handles UTF-8 / Latin-1 encoding, optional BOM stripping,
/// configurable delimiter, and automatic empty-row removal.
///
/// Example:
/// ```dart
/// final loader = CsvLoader('/sdcard/iris.csv');
/// final table  = await loader.load();
/// final schema = table.inferSchema(targetColumn: 'species');
/// ```
class CsvLoader {
  final String path;
  final bool hasHeader;
  final String delimiter;
  final String encoding; // 'utf-8' | 'latin1'

  const CsvLoader(
    this.path, {
    this.hasHeader = true,
    this.delimiter = ',',
    this.encoding = 'utf-8',
  });

  /// Loads and parses the CSV file.
  ///
  /// Throws [CsvFileNotFoundException] if the file does not exist.
  /// Throws [EmptyCsvException] if the file has no parseable rows.
  Future<RawDataTable> load() async {
    final file = File(path);

    if (!await file.exists()) {
      throw CsvFileNotFoundException(path);
    }

    String content;
    try {
      content = await file.readAsString();
    } catch (_) {
      // Fallback: read as bytes → latin1
      final bytes = await file.readAsBytes();
      content = String.fromCharCodes(bytes);
    }

    // Strip UTF-8 BOM if present
    if (content.startsWith('\uFEFF')) {
      content = content.substring(1);
    }

    // Normalise line endings
    content = content.replaceAll('\r\n', '\n').replaceAll('\r', '\n');

    final rows = const CsvToListConverter(
      eol: '\n',
      shouldParseNumbers: true,
    ).convert(content, fieldDelimiter: delimiter);

    // Remove completely empty rows
    final nonEmpty = rows
        .where((r) => r.any((c) => c.toString().trim().isNotEmpty))
        .toList();

    if (nonEmpty.isEmpty) {
      throw EmptyCsvException(path);
    }

    List<String> headers;
    List<List<dynamic>> dataRows;

    if (hasHeader) {
      headers = nonEmpty.first
          .map((e) => e.toString().trim())
          .toList()
          .cast<String>();
      dataRows = nonEmpty.sublist(1);
    } else {
      headers = List.generate(
        nonEmpty.first.length,
        (i) => 'col_$i',
      );
      dataRows = nonEmpty;
    }

    return RawDataTable(
      headers: headers,
      rows: dataRows,
      sourcePath: path,
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RawDataTable
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory representation of a loaded CSV file.
class RawDataTable {
  final List<String> headers;
  final List<List<dynamic>> rows;
  final String sourcePath;

  const RawDataTable({
    required this.headers,
    required this.rows,
    required this.sourcePath,
  });

  int get rowCount => rows.length;
  int get columnCount => headers.length;

  /// Returns the values for a single column.
  List<dynamic> column(String name) {
    final idx = headers.indexOf(name);
    if (idx == -1) throw ColumnNotFoundException(name, headers);
    return rows.map((r) => r[idx]).toList();
  }

  /// Automatically infers a [DataSchema] from the table contents.
  ///
  /// Determines numeric vs categorical columns, output class count,
  /// and class label strings for the target column.
  DataSchema inferSchema({required String targetColumn}) {
    final targetIndex = headers.indexOf(targetColumn);
    if (targetIndex == -1) {
      throw ColumnNotFoundException(targetColumn, headers);
    }

    final columnTypes = <String, ColumnType>{};
    for (int i = 0; i < headers.length; i++) {
      final values = rows.map((r) => r[i]).toList();
      columnTypes[headers[i]] = _inferColumnType(values);
    }

    // Collect unique target values (labels)
    final targetValues =
        rows.map((r) => r[targetIndex].toString().trim()).toSet();
    final sortedLabels = targetValues.toList()..sort();

    return DataSchema(
      headers: headers,
      columnTypes: columnTypes,
      targetColumn: targetColumn,
      targetIndex: targetIndex,
      outputClasses: targetValues.length,
      classLabels: sortedLabels,
    );
  }

  /// Heuristic: if all non-null values parse as double → numeric.
  /// If ≤ 30 unique string values → categorical.
  /// Otherwise → text (ignored during feature extraction).
  static ColumnType _inferColumnType(List<dynamic> values) {
    final nonNull =
        values.where((v) => v != null && v.toString().trim().isNotEmpty);
    if (nonNull.isEmpty) return ColumnType.numeric;

    final allNumeric = nonNull.every(
      (v) => double.tryParse(v.toString()) != null,
    );
    if (allNumeric) return ColumnType.numeric;

    final unique = values.map((v) => v.toString().trim()).toSet();
    if (unique.length <= 30) return ColumnType.categorical;

    return ColumnType.text;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exceptions
// ─────────────────────────────────────────────────────────────────────────────

/// Thrown when the CSV file cannot be found at the given path.
class CsvFileNotFoundException extends OfflineMlException {
  final String csvPath;
  CsvFileNotFoundException(this.csvPath)
      : super('CSV file not found: "$csvPath"');
}

/// Thrown when the CSV file exists but contains no parseable rows.
class EmptyCsvException extends OfflineMlException {
  final String csvPath;
  EmptyCsvException(this.csvPath)
      : super('CSV file is empty: "$csvPath"');
}
