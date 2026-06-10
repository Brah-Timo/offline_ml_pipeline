# Data Pipeline

The data pipeline is responsible for loading raw data, preprocessing it into numeric tensors, and splitting it into training/validation sets.

---

## CsvLoader

`CsvLoader` reads a UTF-8 CSV file into a `RawDataset`.

```dart
final loader = CsvLoader(
  filePath: 'assets/data.csv',
  labelColumn: 'target',      // column name used as label
  delimiter: ',',              // default ','
  hasHeader: true,             // default true
  skipRows: 0,                 // rows to skip after header
);

final dataset = await loader.load();
print('${dataset.rowCount} rows, ${dataset.featureNames.length} features');
```

### Behaviour

- **Header row**: feature column names are extracted from the first row when `hasHeader: true`.
- **Label column**: the label column is separated from the feature columns automatically.
- **Type inference**: numeric strings are parsed as `double`; non-numeric values are kept as `String` for encoding.
- **Error handling**: throws `DataLoadException` on file-not-found or malformed CSV rows.

---

## DataNormalizer

`DataNormalizer` fits a scaling transformation on the training set and applies it to any dataset.

### Z-Score Normalisation

```dart
final normalizer = DataNormalizer.fitZScore(trainDataset);
final normalized  = normalizer.transform(dataset);

// Access scaling parameters
print(normalizer.means);   // Map<String, double>
print(normalizer.stdDevs); // Map<String, double>
```

Applies: `x′ = (x − μ) / σ` per feature column.

### Min-Max Normalisation

```dart
final normalizer = DataNormalizer.fitMinMax(trainDataset, min: 0.0, max: 1.0);
```

Applies: `x′ = (x − min) / (max − min)`.

### Inverse Transform

```dart
final original = normalizer.inverseTransform(normalizedDataset);
```

### Serialisation

`DataNormalizer` implements `toJson()` / `fromJson()` for persistence alongside the exported model:

```dart
final json = normalizer.toJson();
File('normalizer.json').writeAsStringSync(jsonEncode(json));

final loaded = DataNormalizer.fromJson(jsonDecode(...));
```

---

## FeatureEncoder

`FeatureEncoder` handles categorical features.

### One-Hot Encoding

```dart
final encoder = FeatureEncoder.fitOneHot(trainDataset);
final encoded  = encoder.transform(dataset);

print(encoder.categories); // Map<String, List<String>>
```

Each string-valued feature column `col` with categories `[A, B, C]` is expanded into three binary columns: `col_A`, `col_B`, `col_C`.

### Label Encoding

```dart
final encoder = FeatureEncoder.fitLabel(trainDataset);
```

Maps each category to an integer index (0, 1, 2, …). Suitable for tree models; not recommended for neural networks (use one-hot instead).

### Inverse Transform

```dart
final decoded = encoder.inverseTransform(encodedDataset);
```

---

## ProcessedDataset

After normalisation and encoding, data is held in a `ProcessedDataset`:

```dart
ProcessedDataset {
  final List<double> allFeatures;  // flat row-major [N × inputFeatures]
  final List<dynamic> allLabels;   // length N
  final int rowCount;
  final int featureCount;

  // Slice a mini-batch
  DataBatch slice(int start, int end);

  // Access all features/labels
  List<double> get allFeatures;
  List<dynamic> get allLabels;

  // Shuffle rows (in-place, reproducible)
  void shuffle({int? seed});
}
```

### Train/Test Split

```dart
final split = processed.trainTestSplit(
  testFraction: 0.2,
  seed: 42,          // reproducible shuffle
);
// split.train  ProcessedDataset — 80 %
// split.test   ProcessedDataset — 20 %
```

---

## DataBatch

Mini-batches used by `TrainingLoop`:

```dart
DataBatch {
  final List<double> features;  // flat [batchSize × featureCount]
  final List<dynamic> labels;   // length batchSize
  final int size;               // == batchSize
}
```

`ProcessedDataset.slice(start, end)` extracts rows `[start, end)` as a `DataBatch`.

---

## Full Pipeline Example

```dart
// Load
final raw = await CsvLoader(
  filePath: 'data/train.csv',
  labelColumn: 'fraud',
).load();

// Fit transformers on training data
final normalizer = DataNormalizer.fitZScore(raw);
final encoder    = FeatureEncoder.fitOneHot(raw);

// Transform
final processed = normalizer.transform(encoder.transform(raw));

// Split
final split = processed.trainTestSplit(testFraction: 0.15, seed: 0);

print('Train: ${split.train.rowCount} rows');
print('Val:   ${split.test.rowCount}  rows');
print('Features: ${split.train.featureCount}');
```

---

## DataSchema Reference

| Class | Role |
|-------|------|
| `RawDataset` | Raw string-valued rows from CSV |
| `ProcessedDataset` | Float feature matrix + label vector |
| `DataBatch` | Single mini-batch slice |
| `DataNormalizer` | Z-score / min-max scaler |
| `FeatureEncoder` | One-hot / label encoder |
| `TrainTestSplit` | Container for `train` + `test` |
