# Training

This document covers the training loop, session backends, optimizer configuration, loss functions, and early stopping.

---

## TrainingSession

`TrainingSession` is the abstract base class for both training backends. Use the factory method to create one — it automatically selects ORT or Dart fallback:

```dart
final session = await TrainingSession.create(
  spec: spec,
  artifactDir: '/path/to/ort_artifacts',  // ignored by Dart fallback
  optimizerConfig: const OptimizerConfig(
    type: OptimizerType.adam,
    learningRate: 0.001,
  ),
  lossFunction: LossFunction.crossEntropy,
);
```

### Backend selection logic

1. `_OrtBackendSession._init(...)` is attempted.
2. `OrtBindings.instance` calls `DynamicLibrary.open(...)` on the native `.so`/`.dylib`.
3. If this throws (library absent, wrong platform), the factory catches the error and returns a `DartTrainingSession` instead.

### ORT backend (`_OrtBackendSession`)

Wraps the ONNX Runtime On-Device Training C API:

- `createEnv` → `createTrainingSession` using checkpoint + training/eval/optimizer ONNX models.
- `trainStep` → runs the training model forward pass + backward pass + gradient application.
- `optimizerStep` → applies Adam gradient update.
- `evalStep` → runs the eval model (no gradients).
- `exportModelForInferencing` → strips training nodes and writes an inference-only `.onnx`.

### Dart fallback (`DartTrainingSession`)

Backed by `NeuralModel`, a hand-written MLP with:

- Xavier weight initialisation.
- ReLU activations (hidden layers) + Softmax/linear output.
- Adam optimizer (`β₁=0.9`, `β₂=0.999`, `ε=1e-8`).
- Cross-entropy and MSE loss implemented in `LossFunctions`.

---

## TrainingLoop

`TrainingLoop` orchestrates the full epoch/batch cycle:

```dart
final loop = TrainingLoop(
  session: session,
  trainData: split.train,
  valData: split.test,
  epochs: 100,
  batchSize: 32,
  progressNotifier: ProgressNotifier(),
  shuffleEachEpoch: true,
  earlyStopping: EarlyStopping(patience: 10, monitor: 'val_loss'),
  lrSchedule: ExponentialDecaySchedule(decayRate: 0.95),
  initialLr: 0.001,
);

final metrics = await loop.run();
```

### Loop algorithm

```
for epoch in 1..epochs:
  lr = lrSchedule.compute(initialLr, epoch)   // adaptive LR
  shuffle trainData                             // if shuffleEachEpoch
  for batch in mini_batches(trainData):
    loss = session.trainStep(batch)
    check for NaN/Inf → throw TrainingStepException
  epochTrainLoss = mean(batch losses)
  epochValLoss   = session.evalStep(valData)   // if valData not empty
  emit TrainingProgress event
  check early stopping
return _computeFinalMetrics()
```

### EarlyStopping

```dart
const EarlyStopping({
  String monitor = 'val_loss',    // 'val_loss' or 'val_accuracy'
  int patience = 10,              // epochs without improvement
  double minDelta = 1e-4,         // minimum improvement threshold
});
```

Training halts when the monitored metric does not improve by `minDelta` for `patience` consecutive epochs.

---

## OptimizerConfig

```dart
const OptimizerConfig({
  required OptimizerType type,    // adam | sgd | rmsprop
  required double learningRate,
  double beta1 = 0.9,             // Adam β₁
  double beta2 = 0.999,           // Adam β₂
  double epsilon = 1e-8,          // Adam ε
  double momentum = 0.9,          // SGD momentum
  double weightDecay = 0.0,       // L2 regularisation coefficient
});
```

### Learning-Rate Schedules

Implement `LrSchedule` or use the built-in schedules:

| Schedule | Description |
|----------|-------------|
| `ExponentialDecaySchedule` | `lr × decayRate^epoch` |
| `StepDecaySchedule` | halves LR every `dropEvery` epochs |
| `CosineAnnealingSchedule` | cosine annealing between `lrMax` and `lrMin` |
| `WarmupSchedule` | linear warm-up for `warmupEpochs` then decays |

```dart
final schedule = ExponentialDecaySchedule(decayRate: 0.95);
// or
final schedule = CosineAnnealingSchedule(
  lrMax: 0.01, lrMin: 1e-5, cycleLength: 20,
);
```

---

## LossFunctions

All static, called by both backends:

```dart
// Classification
double loss = LossFunctions.crossEntropy(preds, labels);
// preds: List<List<double>> — softmax probabilities per class
// labels: List<int> — class indices

// Regression
double loss = LossFunctions.mse(predValues, trueValues);
double r2   = LossFunctions.rSquared(predValues, trueValues);
double mae  = LossFunctions.mae(predValues, trueValues);
double mape = LossFunctions.mape(predValues, trueValues);
```

The `LossFunction` enum selects which function the session uses internally:

```dart
enum LossFunction {
  crossEntropy,
  binaryCrossEntropy,
  mse,
  mae,
  huber,
}
```

---

## Epoch Records

`TrainingLoop.epochHistory` contains one `EpochRecord` per completed epoch:

```dart
class EpochRecord {
  final int epoch;
  final double trainLoss;
  final double valLoss;
  final DateTime timestamp;
}

// Access after run():
for (final r in loop.epochHistory) {
  print('${r.epoch}: train=${r.trainLoss} val=${r.valLoss}');
}
print('Total duration: ${loop.duration}');
```

---

## Progress Notifications

```dart
final notifier = ProgressNotifier();
notifier.stream.listen((p) {
  print('${p.epoch}/${p.totalEpochs} — '
        'train: ${p.trainLoss.toStringAsFixed(4)} '
        'val: ${p.valLoss.toStringAsFixed(4)} '
        '(${p.percentage}%)');
});
notifier.earlyStopStream.listen((epoch) {
  print('Early stopping at epoch $epoch');
});
```

---

## Accessing Metrics

After `loop.run()` returns:

```dart
final metrics = await loop.run();

// Classifier
print(metrics.accuracy);       // 0.93
print(metrics.f1Score);        // 0.92
print(metrics.confusionMatrix);

// Regressor
print(metrics.rmse);
print(metrics.rSquared);

// Both
print(metrics.finalTrainLoss);
print(metrics.finalValLoss);
print(metrics.totalEpochs);
print(metrics.stoppedEarly);

// Human-readable report
print(metrics.toReport(ModelType.classifier));
```
