import 'dart:io';
import 'package:test/test.dart';
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';

// Helper to write a temp CSV and return a RawDataTable
Future<RawDataTable> _loadCsv(String csv) async {
  final file = await File(
          '${Directory.systemTemp.path}/norm_test_${DateTime.now().millisecondsSinceEpoch}.csv')
      .create();
  await file.writeAsString(csv);
  return CsvLoader(file.path).load();
}

void main() {
  group('DataNormalizer — MinMax', () {
    test('scales numeric column to [0, 1]', () async {
      final table = await _loadCsv('x,y,label\n0,10,a\n5,20,b\n10,30,c\n');
      final schema = table.inferSchema(targetColumn: 'label');

      final norm = DataNormalizer(strategy: NormalizationStrategy.minMax);
      final result = norm.fitTransform(table, schema);

      final xVals = result.rows.map((r) {
        return double.parse(r[result.headers.indexOf('x')].toString());
      }).toList();

      expect(xVals, everyElement(greaterThanOrEqualTo(0.0)));
      expect(xVals, everyElement(lessThanOrEqualTo(1.0)));
      expect(xVals.first, closeTo(0.0, 1e-6));
      expect(xVals.last, closeTo(1.0, 1e-6));
    });

    test('does not transform target column', () async {
      final table = await _loadCsv('x,label\n0.0,yes\n1.0,no\n');
      final schema = table.inferSchema(targetColumn: 'label');

      final norm = DataNormalizer(strategy: NormalizationStrategy.minMax);
      final result = norm.fitTransform(table, schema);

      final labelIdx = result.headers.indexOf('label');
      final labels = result.rows.map((r) => r[labelIdx].toString()).toList();
      expect(labels, equals(['yes', 'no']));
    });

    test('serialize / fromJson round-trip', () async {
      final table = await _loadCsv('a,b,label\n1,2,x\n3,4,y\n5,6,z\n');
      final schema = table.inferSchema(targetColumn: 'label');

      final norm = DataNormalizer(strategy: NormalizationStrategy.minMax);
      norm.fitTransform(table, schema);

      final json = norm.serialize();
      final restored = DataNormalizer.fromJson(json);

      // normalizeValue should produce the same result
      final orig = norm.normalizeValue(3.0, 'a');
      final rest = restored.normalizeValue(3.0, 'a');
      expect(orig, closeTo(rest, 1e-9));
    });
  });

  group('DataNormalizer — ZScore', () {
    test('produces approximately zero mean', () async {
      final table = await _loadCsv(
          'v,label\n10,a\n20,b\n30,c\n40,d\n50,e\n');
      final schema = table.inferSchema(targetColumn: 'label');

      final norm = DataNormalizer(strategy: NormalizationStrategy.zScore);
      final result = norm.fitTransform(table, schema);

      final vals = result.rows.map((r) {
        return double.parse(r[result.headers.indexOf('v')].toString());
      }).toList();

      final mean = vals.reduce((a, b) => a + b) / vals.length;
      expect(mean.abs(), lessThan(1e-6));
    });
  });

  group('DataNormalizer — None', () {
    test('passes values through unchanged', () async {
      final table = await _loadCsv('x,label\n42.5,a\n99.9,b\n');
      final schema = table.inferSchema(targetColumn: 'label');

      final norm = DataNormalizer(strategy: NormalizationStrategy.none);
      final result = norm.fitTransform(table, schema);

      final xIdx = result.headers.indexOf('x');
      expect(double.parse(result.rows[0][xIdx].toString()), closeTo(42.5, 1e-6));
    });
  });
}
