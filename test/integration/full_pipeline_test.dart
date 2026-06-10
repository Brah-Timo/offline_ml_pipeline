// ignore_for_file: lines_longer_than_80_chars

import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';

String get _irisPath => p.join(
      Directory.current.path,
      'test', 'integration', 'fixtures', 'iris.csv',
    );

void main() {
  // ── Full pipeline integration tests ──────────────────────────────────────
  //
  // These tests run the Dart-fallback training backend end-to-end.
  // ORT native tests require device hardware and are run on CI only.

  group('Full Pipeline (Dart fallback)', () {
    test('trains classifier on iris dataset', () async {
      final outputDir = await Directory.systemTemp.createTemp('oml_test_');

      final pipeline = MlPipeline(
        config: PipelineConfig(
          csvPath: _irisPath,
          targetColumn: 'species',
          modelType: ModelType.classifier,
          epochs: 30,
          batchSize: 10,
          optimizerConfig: const OptimizerConfig.adam(learningRate: 0.01),
          outputDirectory: outputDir.path,
          // Use minRatio so all three classes appear in every split
          trainRatio: 0.7,
          valRatio: 0.15,
          testRatio: 0.15,
        ),
      );

      final progressEvents = <TrainingProgress>[];
      pipeline.progressStream.listen(progressEvents.add);

      final result = await pipeline.train();

      // ── Assertions ──────────────────────────────────────────────────────

      // Model file exists
      expect(result.modelExists, isTrue,
          reason: 'Model file should be written to disk');

      // Progress events received
      expect(progressEvents, isNotEmpty,
          reason: 'Should receive at least one progress event');
      expect(progressEvents.first.epoch, equals(1));
      expect(progressEvents.last.percentage, equals(100));

      // Schema has correct structure
      expect(result.schema.featureCount, equals(4),
          reason: 'Iris has 4 numeric features');
      expect(result.schema.outputClasses, equals(3),
          reason: 'Iris has 3 classes');

      // Normaliser and encoder state persisted
      expect(result.normalizerState, isNotEmpty);
      expect(result.encoderState, isNotEmpty);

      // Training took some time
      expect(result.trainingDuration.inMilliseconds, greaterThan(0));

      // Epoch history recorded
      expect(result.epochHistory.length, equals(30));

      // Metrics not null (Dart fallback computes them)
      expect(result.metrics.finalTrainLoss, isNotNaN);
      expect(result.metrics.finalTrainLoss, isNot(equals(double.infinity)));

      print(result);
      print(result.metrics.toReport(ModelType.classifier));

      // Cleanup
      await outputDir.delete(recursive: true);
    }, timeout: const Timeout(Duration(minutes: 3)));

    test('trains regressor', () async {
      final dir = await Directory.systemTemp.createTemp('oml_reg_');

      // Build a simple regression CSV
      final csvFile = File('${dir.path}/regression.csv');
      final lines = StringBuffer('x1,x2,x3,target\n');
      for (int i = 0; i < 80; i++) {
        lines.writeln('${i * 0.1},${i * 0.2},${i * 0.3},${i * 0.5}');
      }
      await csvFile.writeAsString(lines.toString());

      final pipeline = MlPipeline(
        config: PipelineConfig(
          csvPath: csvFile.path,
          targetColumn: 'target',
          modelType: ModelType.regressor,
          epochs: 20,
          batchSize: 16,
          lossFunction: LossFunction.mse,
          outputDirectory: dir.path,
        ),
      );

      final result = await pipeline.train();

      expect(result.modelExists, isTrue);
      expect(result.metrics.finalTrainLoss, isNotNaN);

      await dir.delete(recursive: true);
    }, timeout: const Timeout(Duration(minutes: 2)));

    test('throws CsvFileNotFoundException for missing file', () {
      final pipeline = MlPipeline(
        config: PipelineConfig(
          csvPath: '/no/such/file.csv',
          targetColumn: 'label',
          modelType: ModelType.classifier,
        ),
      );

      expect(
        pipeline.train(),
        throwsA(isA<CsvFileNotFoundException>()),
      );
    });
  });
}
