# Testing

## Running the Test Suite

```bash
# All tests
flutter test

# Unit tests only
flutter test test/unit/

# Integration tests only
flutter test test/integration/

# A specific file
flutter test test/unit/training_metrics_test.dart

# With verbose output
flutter test --reporter expanded
```

---

## Test Layout

```
test/
├── unit/
│   ├── csv_loader_test.dart          # CsvLoader parsing, edge cases
│   ├── data_normalizer_test.dart     # Z-score, min-max, inverse transform
│   └── training_metrics_test.dart   # LossFunctions, MetricsCalculator, TrainingMetrics
└── integration/
    └── full_pipeline_test.dart       # CSV → preprocess → train → export end-to-end
```

---

## Unit Tests

### `csv_loader_test.dart`

Tests cover:
- Parses standard comma-separated CSV with header.
- Correctly separates label column from feature columns.
- Handles quoted fields containing commas.
- Handles empty/missing rows.
- Throws `DataLoadException` for non-existent files.
- Supports custom delimiter (`;`, `\t`).

### `data_normalizer_test.dart`

Tests cover:
- Z-score: mean=0, std=1 after transform.
- Min-max: values in `[0, 1]` after transform.
- `inverseTransform` round-trips original values.
- `fromJson(toJson())` round-trip preserves scaling parameters.
- Handles zero-variance columns gracefully (does not divide by zero).

### `training_metrics_test.dart`

Tests cover:
- `LossFunctions.crossEntropy`: perfect predictions → loss ≈ 0.
- `LossFunctions.mse`: identical predictions → 0.
- `LossFunctions.rSquared`: mean predictions → R² = 0.
- `MetricsCalculator.classification`: perfect predictions → accuracy = 1.0, F1 = 1.0.
- `MetricsCalculator.classification`: all wrong → accuracy = 0.0.
- `MetricsCalculator.classification`: confusion matrix diagonal equals TP counts.
- `MetricsCalculator.regression`: perfect fit → RMSE = 0, R² = 1.0.
- `TrainingMetrics.toReport(classifier)`: contains "Accuracy", "F1".
- `TrainingMetrics.toReport(regressor)`: contains "RMSE", "R²".

---

## Integration Tests

### `full_pipeline_test.dart`

End-to-end test using the **Dart fallback backend** (ORT is not available in CI):

1. Generates a synthetic `iris`-style dataset (150 rows, 4 features, 3 classes).
2. Writes it to a temp CSV.
3. Runs `CsvLoader.load()`.
4. Fits `DataNormalizer` and `FeatureEncoder`.
5. Calls `ProcessedDataset.trainTestSplit()`.
6. Creates a `DartTrainingSession`.
7. Runs `TrainingLoop` for 10 epochs.
8. Asserts `metrics.finalTrainLoss` is finite.
9. Calls `ModelExporter.exportToTflite()`.
10. Asserts output file exists and has non-zero size.

---

## Writing New Tests

### Unit test template

```dart
import 'package:test/test.dart';
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';

void main() {
  group('MyClass', () {
    late MyClass sut;

    setUp(() {
      sut = MyClass(param: 'value');
    });

    tearDown(() {
      sut.dispose();
    });

    test('does something correctly', () {
      final result = sut.doSomething();
      expect(result, equals(expectedValue));
    });

    test('throws on invalid input', () {
      expect(() => sut.doSomething(null), throwsA(isA<DataLoadException>()));
    });
  });
}
```

### Import guidelines

Always import the **barrel export** only:

```dart
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';
```

Do **not** add a second direct `lib/src/...` import for the same symbols — it creates duplicate type ambiguity errors.

### Mocking `TrainingSession`

Use `mockito` to mock the abstract `TrainingSession`:

```dart
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';
import 'my_widget_test.mocks.dart';

@GenerateMocks([TrainingSession])
void main() {
  test('loop calls trainStep per batch', () async {
    final mockSession = MockTrainingSession();
    when(mockSession.spec).thenReturn(spec);
    when(mockSession.trainStep(
      flatInputs: anyNamed('flatInputs'),
      labels: anyNamed('labels'),
      batchSize: anyNamed('batchSize'),
    )).thenAnswer((_) async => 0.5);
    when(mockSession.evalStep(
      flatInputs: anyNamed('flatInputs'),
      labels: anyNamed('labels'),
      batchSize: anyNamed('batchSize'),
    )).thenAnswer((_) async => 0.4);
    when(mockSession.predict(
      flatInputs: anyNamed('flatInputs'),
      n: anyNamed('n'),
    )).thenAnswer((_) async => [[0.8, 0.2]]);
    when(mockSession.backend).thenReturn(TrainingBackend.dartFallback);

    final metrics = await TrainingLoop(
      session: mockSession,
      ...
    ).run();

    verify(mockSession.trainStep(...)).called(greaterThan(0));
  });
}
```

Generate mocks: `flutter pub run build_runner build`.

---

## CI / CD

Recommended GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: subosito/flutter-action@v2
        with:
          flutter-version: '3.x'
      - run: flutter pub get
      - run: flutter test --reporter expanded
      - run: flutter analyze
```

The ORT backend is not tested in CI (no native library). All integration tests use `DartTrainingSession` via the automatic fallback.

---

## Coverage

Generate an LCOV coverage report:

```bash
flutter test --coverage
genhtml coverage/lcov.info -o coverage/html
open coverage/html/index.html
```

Key coverage targets:
- `data/` — >95%
- `training/` — >90%
- `export/` — >85%
- `ffi/` — excluded (requires native library)
