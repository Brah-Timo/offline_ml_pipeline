import 'package:test/test.dart';
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';

void main() {
  group('LossFunctions', () {
    test('crossEntropy: perfect predictions → loss near 0', () {
      final preds = [
        [0.99, 0.01],
        [0.01, 0.99],
      ];
      final labels = [0, 1];
      final loss = LossFunctions.crossEntropy(preds, labels);
      expect(loss, lessThan(0.02));
    });

    test('crossEntropy: random predictions → higher loss', () {
      final preds = [
        [0.5, 0.5],
        [0.5, 0.5],
      ];
      final labels = [0, 1];
      final loss = LossFunctions.crossEntropy(preds, labels);
      expect(loss, greaterThan(0.6));
    });

    test('mse: identical predictions → 0', () {
      final loss = LossFunctions.mse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
      expect(loss, closeTo(0.0, 1e-9));
    });

    test('mse: off by 1 each → 1.0', () {
      final loss = LossFunctions.mse([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]);
      expect(loss, closeTo(1.0, 1e-9));
    });

    test('rSquared: perfect fit → 1.0', () {
      final r2 = LossFunctions.rSquared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
      expect(r2, closeTo(1.0, 1e-9));
    });

    test('rSquared: mean prediction → 0.0', () {
      // If we always predict the mean, R² = 0
      final targets = [1.0, 2.0, 3.0];
      final mean = 2.0;
      final preds = List.filled(3, mean);
      final r2 = LossFunctions.rSquared(preds, targets);
      expect(r2.abs(), lessThan(1e-6));
    });
  });

  group('MetricsCalculator.classification', () {
    test('perfect classification → accuracy 1.0', () {
      final preds  = [0, 1, 2, 0, 1, 2];
      final labels = [0, 1, 2, 0, 1, 2];
      final m = MetricsCalculator.classification(
        preds: preds, labels: labels, numClasses: 3,
      );
      expect(m.accuracy, closeTo(1.0, 1e-9));
      expect(m.f1Score, closeTo(1.0, 1e-6));
    });

    test('all wrong → accuracy 0.0', () {
      final preds  = [1, 2, 0];
      final labels = [0, 1, 2];
      final m = MetricsCalculator.classification(
        preds: preds, labels: labels, numClasses: 3,
      );
      expect(m.accuracy, closeTo(0.0, 1e-9));
    });

    test('confusion matrix diagonal is TP count', () {
      final preds  = [0, 0, 1, 1, 2, 2];
      final labels = [0, 0, 1, 1, 2, 2];
      final m = MetricsCalculator.classification(
        preds: preds, labels: labels, numClasses: 3,
      );
      // Perfect → all off-diagonal = 0
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          if (i == j) {
            expect(m.confusionMatrix[i][j], equals(2));
          } else {
            expect(m.confusionMatrix[i][j], equals(0));
          }
        }
      }
    });
  });

  group('MetricsCalculator.regression', () {
    test('perfect regression → rmse 0, r2 1', () {
      final m = MetricsCalculator.regression(
        preds: [1.0, 2.0, 3.0], labels: [1.0, 2.0, 3.0],
      );
      expect(m.rmse, closeTo(0.0, 1e-9));
      expect(m.rSquared, closeTo(1.0, 1e-9));
    });
  });

  group('TrainingMetrics.toReport', () {
    test('classifier report contains Accuracy', () {
      final m = TrainingMetrics(
        accuracy: 0.92,
        f1Score: 0.91,
        precision: 0.90,
        recall: 0.93,
        finalTrainLoss: 0.15,
        finalValLoss: 0.18,
        totalEpochs: 50,
      );
      final report = m.toReport(ModelType.classifier);
      expect(report, contains('Accuracy'));
      expect(report, contains('92.00'));
      expect(report, contains('F1'));
    });

    test('regressor report contains RMSE and R²', () {
      final m = TrainingMetrics(
        rmse: 0.05,
        mae: 0.04,
        rSquared: 0.95,
        finalTrainLoss: 0.002,
        finalValLoss: 0.003,
        totalEpochs: 100,
      );
      final report = m.toReport(ModelType.regressor);
      expect(report, contains('RMSE'));
      expect(report, contains('R²'));
    });
  });
}
