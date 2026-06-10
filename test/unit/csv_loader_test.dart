import 'dart:io';
import 'package:test/test.dart';
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';

void main() {
  group('CsvLoader', () {
    late Directory tempDir;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('offline_ml_test_');
    });

    tearDown(() async {
      await tempDir.delete(recursive: true);
    });

    test('loads a simple CSV with header', () async {
      final file = File('${tempDir.path}/test.csv');
      await file.writeAsString('a,b,c\n1,2,3\n4,5,6\n');

      final loader = CsvLoader(file.path);
      final table = await loader.load();

      expect(table.headers, equals(['a', 'b', 'c']));
      expect(table.rowCount, equals(2));
      expect(table.columnCount, equals(3));
    });

    test('loads a CSV without header', () async {
      final file = File('${tempDir.path}/noheader.csv');
      await file.writeAsString('1,2,3\n4,5,6\n');

      final loader = CsvLoader(file.path, hasHeader: false);
      final table = await loader.load();

      expect(table.headers, equals(['col_0', 'col_1', 'col_2']));
      expect(table.rowCount, equals(2));
    });

    test('strips UTF-8 BOM', () async {
      final file = File('${tempDir.path}/bom.csv');
      final bom = [0xEF, 0xBB, 0xBF]; // UTF-8 BOM bytes
      await file.writeAsBytes([...bom, ...'a,b\n1,2\n'.codeUnits]);

      final loader = CsvLoader(file.path);
      final table = await loader.load();

      expect(table.headers.first, equals('a'));
    });

    test('removes empty rows', () async {
      final file = File('${tempDir.path}/empties.csv');
      await file.writeAsString('x,y\n1,2\n\n\n3,4\n');

      final loader = CsvLoader(file.path);
      final table = await loader.load();

      expect(table.rowCount, equals(2));
    });

    test('throws CsvFileNotFoundException for missing file', () async {
      final loader = CsvLoader('/nonexistent/path/file.csv');
      expect(loader.load(), throwsA(isA<CsvFileNotFoundException>()));
    });

    test('throws EmptyCsvException for empty file', () async {
      final file = File('${tempDir.path}/empty.csv');
      await file.writeAsString('');

      final loader = CsvLoader(file.path);
      expect(loader.load(), throwsA(isA<EmptyCsvException>()));
    });

    test('supports tab delimiter', () async {
      final file = File('${tempDir.path}/tab.tsv');
      await file.writeAsString('x\ty\n1\t2\n3\t4\n');

      final loader = CsvLoader(file.path, delimiter: '\t');
      final table = await loader.load();

      expect(table.headers, equals(['x', 'y']));
      expect(table.rowCount, equals(2));
    });
  });

  group('RawDataTable.inferSchema', () {
    test('detects numeric and categorical columns', () async {
      final file = await File(
              '${Directory.systemTemp.path}/schema_test.csv')
          .create();
      await file.writeAsString(
          'sepal_length,sepal_width,species\n'
          '5.1,3.5,setosa\n'
          '4.9,3.0,versicolor\n'
          '6.3,3.3,virginica\n');

      final table = await CsvLoader(file.path).load();
      final schema = table.inferSchema(targetColumn: 'species');

      expect(schema.columnTypes['sepal_length'], equals(ColumnType.numeric));
      expect(schema.columnTypes['sepal_width'], equals(ColumnType.numeric));
      expect(schema.columnTypes['species'], equals(ColumnType.categorical));
      expect(schema.outputClasses, equals(3));
      expect(schema.classLabels, containsAll(['setosa', 'versicolor', 'virginica']));
    });

    test('throws ColumnNotFoundException for missing target', () async {
      final file = await File(
              '${Directory.systemTemp.path}/schema_missing.csv')
          .create();
      await file.writeAsString('a,b\n1,2\n');

      final table = await CsvLoader(file.path).load();

      expect(
        () => table.inferSchema(targetColumn: 'nonexistent'),
        throwsA(isA<ColumnNotFoundException>()),
      );
    });
  });
}
