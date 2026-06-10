# Performance Guide

## Backend Comparison

| Metric | ORT backend | Dart fallback |
|--------|------------|---------------|
| Training speed | Fast (SIMD + parallelism via ORT native) | Moderate (single-threaded Dart) |
| Inference latency | Very fast (ORT optimised kernels) | Moderate |
| Memory efficiency | Managed by ORT allocator | Dart GC |
| Platform support | Android, iOS, Linux, Windows, macOS | All (including web via compilation) |
| Setup complexity | Requires native library packaging | Zero native deps |
| Model export | Full ONNX inference graph | JSON + pseudo-TFLite |

**Recommendation**: Use ORT backend for production and Dart fallback for CI testing, web targets, or devices without native library support.

---

## ORT Backend Performance

### Native Library Selection

ORT On-Device Training uses the full ORT training library. To optimise for mobile:

- **Android**: Use `libonnxruntime_training.so` built with NNAPI delegate enabled.
- **iOS**: Use `OnnxRuntime` CocoaPod with CoreML delegate.
- **Reduce binary size**: Strip unused op kernels using `--target_ops` during ONNX Runtime build.

### Batch Size Tuning

Larger batches = better GPU/SIMD utilisation but more memory:

```dart
// Profile different batch sizes
for (final bs in [16, 32, 64, 128]) {
  final sw = Stopwatch()..start();
  final loop = TrainingLoop(
    session: session, batchSize: bs, epochs: 5, ...
  );
  await loop.run();
  print('batchSize=$bs → ${sw.elapsedMilliseconds}ms');
}
```

Typical sweet spots:
- Mobile CPU: `16–32`
- Desktop CPU: `64–128`
- GPU / NNAPI: `128–256`

### Quantisation Impact

| quantizationMode | Load time | Inference time | Memory |
|-----------------|-----------|----------------|--------|
| `none` | baseline | baseline | 100% |
| `float16` | -5% | -20% | 50% |
| `dynamicRange` | -3% | -30% | 25% |
| `int8` | -5% | -40% | 25% |

---

## Dart Fallback Performance

### NeuralModel Forward Pass

The pure-Dart MLP does matrix multiplication via nested loops. Expected throughput (single thread, ~2 GHz device):

| Architecture | Batch size 32 | Inference (N=1) |
|-------------|---------------|-----------------|
| [10, 32, 16, 2] | ~2 ms/step | ~0.1 ms |
| [50, 128, 64, 10] | ~8 ms/step | ~0.3 ms |
| [200, 256, 128, 5] | ~30 ms/step | ~1 ms |

### Reducing Dart Overhead

- Keep `layerSizes` small: each hidden layer doubles computation.
- Prefer fewer, wider layers over many narrow ones.
- Use `batchSize: 64+` to amortise per-call overhead.

---

## Data Pipeline Performance

### CsvLoader

- For large CSV files (>100 MB), stream parsing line-by-line to avoid peak memory:

```dart
// Stream approach (planned for v0.2.0)
await for (final row in loader.loadStream()) {
  pipeline.feed(row);
}
```

- Current `load()` reads the entire file into memory — suitable for datasets up to ~50 MB.

### Normalisation and Encoding

Both `DataNormalizer.transform()` and `FeatureEncoder.transform()` are synchronous O(N×F) operations. For large datasets, consider running them in an `Isolate`:

```dart
final processed = await Isolate.run(() async {
  return normalizer.transform(encoder.transform(raw));
});
```

### ProcessedDataset Shuffle

`ProcessedDataset.shuffle()` uses Fisher-Yates in-place — O(N) time, O(1) extra memory.

---

## Memory Optimisation

### Reuse DataBatch Buffers

Mini-batch slices are new `List<double>` instances. For memory-constrained devices, pre-allocate and reuse:

```dart
// Planned API in v0.2.0
final batchBuffer = Float64List(batchSize * featureCount);
for (int start = 0; start < n; start += batchSize) {
  trainData.sliceInto(start, end, batchBuffer);
  await session.trainStep(flatInputs: batchBuffer, ...);
}
```

### ORT Arena Lifetime

The `Arena` used in FFI calls is scoped to the `using()` block. Native tensors (`OrtValue`) are manually released with `_ort.releaseValue(tensor)` immediately after use — do not hold references across batches.

---

## Profiling

Add epoch timing with `EpochRecord.timestamp`:

```dart
final metrics = await loop.run();
for (int i = 1; i < loop.epochHistory.length; i++) {
  final dt = loop.epochHistory[i].timestamp
      .difference(loop.epochHistory[i - 1].timestamp);
  print('Epoch ${loop.epochHistory[i].epoch}: ${dt.inMilliseconds}ms');
}
print('Total: ${loop.duration}');
```

For line-level profiling on the Dart fallback, use the Dart Observatory (`dart run --observe`).

---

## Training Convergence Tips

| Issue | Symptom | Remedy |
|-------|---------|--------|
| Learning rate too high | NaN loss immediately | Reduce `learningRate` by 10× |
| Learning rate too low | Loss barely moves | Increase `learningRate` or add LR schedule |
| Overfitting | Train loss low, val loss high | Add `weightDecay`, reduce `epochs`, use `EarlyStopping` |
| Underfitting | Both losses high | Increase `layerSizes`, more `epochs`, higher `learningRate` |
| Slow convergence | Linear descent | Use `adam` over `sgd`, tune `beta1`/`beta2` |
| Gradient explosion | Loss spikes or NaN | Add gradient clipping (planned v0.2.0), reduce LR |
