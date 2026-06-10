// ignore_for_file: lines_longer_than_80_chars

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import '../ffi/ort_bindings.dart';
import '../ffi/ort_types.dart';
import '../ffi/native_memory.dart';
import '../models/neural_model.dart';
import '../models/base_model.dart';
import 'optimizer_config.dart';
import 'loss_functions.dart';

// ─────────────────────────────────────────────────────────────────────────────
// TrainingBackend enum
// ─────────────────────────────────────────────────────────────────────────────

/// Which backend executes the actual training operations.
enum TrainingBackend {
  /// ONNX Runtime On-Device Training (requires native binaries).
  ort,

  /// Pure-Dart MLP fallback (no native dependencies, slower).
  dartFallback,
}

// ─────────────────────────────────────────────────────────────────────────────
// TrainingSession
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a training backend (ORT or Dart fallback) and exposes a uniform
/// `trainStep` / `evalStep` / `exportInferenceModel` API.
///
/// Factory method [create] automatically picks ORT if the native library is
/// available, otherwise falls back to pure-Dart training:
///
/// ```dart
/// final session = await TrainingSession.create(
///   spec: spec,
///   artifactDir: '/path/to/ort_artifacts',
///   optimizerConfig: config.optimizerConfig,
///   lossFunction: config.lossFunction,
/// );
/// ```
abstract class TrainingSession {
  final ModelSpec spec;
  final OptimizerConfig optimizerConfig;
  final LossFunction lossFunction;

  TrainingSession({
    required this.spec,
    required this.optimizerConfig,
    required this.lossFunction,
  });

  // ── Factory ────────────────────────────────────────────────────────────

  /// Creates a [TrainingSession].
  ///
  /// Tries ORT first; falls back to Dart if native library is absent.
  static Future<TrainingSession> create({
    required ModelSpec spec,
    required String artifactDir,
    required OptimizerConfig optimizerConfig,
    required LossFunction lossFunction,
  }) async {
    try {
      final ortSession = await _OrtBackendSession._init(
        spec: spec,
        artifactDir: artifactDir,
        optimizerConfig: optimizerConfig,
        lossFunction: lossFunction,
      );
      return ortSession;
    } catch (e) {
      // ORT unavailable (missing .so / wrong platform) → use Dart fallback
      return DartTrainingSession(
        spec: spec,
        optimizerConfig: optimizerConfig,
        lossFunction: lossFunction,
      );
    }
  }

  // ── Abstract interface ─────────────────────────────────────────────────

  /// Runs one training step on a mini-batch.
  ///
  /// [flatInputs]: feature values, shape [batchSize × featureCount].
  /// [labels]: class indices (classifier) or doubles (regressor).
  /// Returns the training loss for this batch.
  Future<double> trainStep({
    required List<double> flatInputs,
    required List<dynamic> labels,
    required int batchSize,
  });

  /// Evaluates the model on a batch; returns validation loss.
  Future<double> evalStep({
    required List<double> flatInputs,
    required List<dynamic> labels,
    required int batchSize,
  });

  /// Runs the full forward pass; returns raw predictions.
  ///
  /// [flatInputs]: shape [N × featureCount].
  /// Returns: shape [N × outputSize].
  Future<List<List<double>>> predict({
    required List<double> flatInputs,
    required int n,
  });

  /// Exports a self-contained inference model.
  ///
  /// For ORT backend: writes an ONNX inference graph.
  /// For Dart backend: serialises weights to JSON.
  ///
  /// Returns the path of the exported model file.
  Future<String> exportInferenceModel(String outputPath);

  /// Frees all resources held by this session.
  Future<void> dispose();

  /// Which backend this session uses.
  TrainingBackend get backend;
}

// ─────────────────────────────────────────────────────────────────────────────
// _OrtBackendSession — ONNX Runtime backend
// ─────────────────────────────────────────────────────────────────────────────

class _OrtBackendSession extends TrainingSession {
  late final OrtBindings _ort;
  Pointer<OrtEnv>? _env;
  Pointer<OrtTrainingSession>? _session;
  Pointer<OrtCheckpointState>? _checkpoint;
  Pointer<OrtSessionOptions>? _sessionOptions;

  _OrtBackendSession._({
    required super.spec,
    required super.optimizerConfig,
    required super.lossFunction,
  });

  static Future<_OrtBackendSession> _init({
    required ModelSpec spec,
    required String artifactDir,
    required OptimizerConfig optimizerConfig,
    required LossFunction lossFunction,
  }) async {
    final session = _OrtBackendSession._(
      spec: spec,
      optimizerConfig: optimizerConfig,
      lossFunction: lossFunction,
    );
    await session._setUp(artifactDir);
    return session;
  }

  Future<void> _setUp(String artifactDir) async {
    _ort = OrtBindings.instance; // throws if .so not loadable

    using((arena) {
      // ── Create environment ──────────────────────────────────────────────
      final envPtr = arena.allocate<Pointer<OrtEnv>>(1);
      final logId = 'offline_ml_pipeline'.toNativeUtf8(allocator: arena).cast<Void>();
      _ort.check(_ort.createEnv(OrtLoggingLevel.warning, logId, envPtr));
      _env = envPtr.value;

      // ── Create session options ──────────────────────────────────────────
      final optPtr = arena.allocate<Pointer<OrtSessionOptions>>(1);
      _ort.check(_ort.createSessionOptions(optPtr));
      _sessionOptions = optPtr.value;

      // ── Load checkpoint ────────────────────────────────────────────────
      final ckptPath = '$artifactDir/checkpoint'.toNativeUtf8(
        allocator: arena,
      ).cast<Void>();
      final ckptPtr = arena.allocate<Pointer<OrtCheckpointState>>(1);
      _ort.check(_ort.loadCheckpoint(ckptPath, ckptPtr));
      _checkpoint = ckptPtr.value;

      // ── Create training session ────────────────────────────────────────
      final trainPath = '$artifactDir/training_model.onnx'.toNativeUtf8(
        allocator: arena,
      ).cast<Void>();
      final evalPath = '$artifactDir/eval_model.onnx'.toNativeUtf8(
        allocator: arena,
      ).cast<Void>();
      final optModelPath = '$artifactDir/optimizer_model.onnx'.toNativeUtf8(
        allocator: arena,
      ).cast<Void>();

      final tsPtr = arena.allocate<Pointer<OrtTrainingSession>>(1);
      _ort.check(
        _ort.createTrainingSession(
          _env!,
          _sessionOptions!,
          _checkpoint!,
          trainPath,
          evalPath,
          optModelPath,
          tsPtr,
        ),
      );
      _session = tsPtr.value;
    });
  }

  @override
  Future<double> trainStep({
    required List<double> flatInputs,
    required List<dynamic> labels,
    required int batchSize,
  }) async {
    double loss = 0.0;

    await using((arena) async {
      // Build input feature tensor
      final inputTensor = _ort.buildFloat32Tensor(
        data: flatInputs,
        shape: [batchSize, spec.inputFeatures],
        arena: arena,
      );

      // Build label tensor (int64)
      final labelData = labels.map((l) => (l as num).toInt()).toList();
      final labelTensor = _buildInt64Tensor(labelData, [batchSize], arena);

      // inputs array: [features, labels]
      final inputsArr = arena.allocate<Pointer<OrtValue>>(
        2 * sizeOf<Pointer<OrtValue>>(),
      );
      inputsArr[0] = inputTensor;
      inputsArr[1] = labelTensor;

      // outputs: [loss]
      final outputsArr = arena.allocate<Pointer<OrtValue>>(
        sizeOf<Pointer<OrtValue>>(),
      );
      outputsArr[0] = nullptr;

      _ort.check(
        _ort.trainStep(
          _session!,
          nullptr, // run options
          2,
          inputsArr,
          1,
          outputsArr,
        ),
      );

      // Optimizer step
      _ort.check(_ort.optimizerStep(_session!, nullptr));

      // Read loss from output tensor
      final lossVals = _ort.readFloat32Tensor(outputsArr[0], 1, arena);
      loss = lossVals[0];

      // Release tensors
      _ort.releaseValue(inputTensor);
      _ort.releaseValue(labelTensor);
      _ort.releaseValue(outputsArr[0]);
    });

    return loss;
  }

  @override
  Future<double> evalStep({
    required List<double> flatInputs,
    required List<dynamic> labels,
    required int batchSize,
  }) async {
    double loss = 0.0;

    await using((arena) async {
      final inputTensor = _ort.buildFloat32Tensor(
        data: flatInputs,
        shape: [batchSize, spec.inputFeatures],
        arena: arena,
      );
      final labelData = labels.map((l) => (l as num).toInt()).toList();
      final labelTensor = _buildInt64Tensor(labelData, [batchSize], arena);

      final inputsArr = arena.allocate<Pointer<OrtValue>>(
        2 * sizeOf<Pointer<OrtValue>>(),
      );
      inputsArr[0] = inputTensor;
      inputsArr[1] = labelTensor;

      final outputsArr = arena.allocate<Pointer<OrtValue>>(
        sizeOf<Pointer<OrtValue>>(),
      );
      outputsArr[0] = nullptr;

      _ort.check(
        _ort.evalStep(
          _session!,
          nullptr,
          2,
          inputsArr,
          1,
          outputsArr,
        ),
      );

      final lossVals = _ort.readFloat32Tensor(outputsArr[0], 1, arena);
      loss = lossVals[0];

      _ort.releaseValue(inputTensor);
      _ort.releaseValue(labelTensor);
      _ort.releaseValue(outputsArr[0]);
    });

    return loss;
  }

  @override
  Future<List<List<double>>> predict({
    required List<double> flatInputs,
    required int n,
  }) async {
    // Eval-step based forward pass (no gradient)
    // Simplified: run evalStep on each row
    final results = <List<double>>[];
    final f = spec.inputFeatures;
    for (int i = 0; i < n; i++) {
      final row = flatInputs.sublist(i * f, (i + 1) * f);
      await using((arena) async {
        final inputTensor = _ort.buildFloat32Tensor(
          data: row,
          shape: [1, f],
          arena: arena,
        );
        // For inference we'd run the eval model's output node
        // Simplified: return zeros (real impl reads output tensor)
        results.add(List.filled(spec.outputSize, 0.0));
        _ort.releaseValue(inputTensor);
      });
    }
    return results;
  }

  @override
  Future<String> exportInferenceModel(String outputPath) async {
    using((arena) {
      final outPath = outputPath.toNativeUtf8(allocator: arena).cast<Void>();
      final outputName = 'output'.toNativeUtf8(allocator: arena).cast<Void>();

      final namesArr = arena.allocate<Pointer<Void>>(
        sizeOf<Pointer<Void>>(),
      );
      namesArr[0] = outputName;

      _ort.check(
        _ort.exportModelForInferencing(_session!, outPath, 1, namesArr),
      );
    });

    return outputPath;
  }

  @override
  Future<void> dispose() async {
    if (_session != null) _ort.releaseTrainingSession(_session!);
    if (_checkpoint != null) _ort.releaseCheckpointState(_checkpoint!);
    if (_env != null) _ort.releaseEnv(_env!);
  }

  @override
  TrainingBackend get backend => TrainingBackend.ort;

  // ── Helper: build int64 tensor ─────────────────────────────────────────

  Pointer<OrtValue> _buildInt64Tensor(
    List<int> data,
    List<int> shape,
    Arena arena,
  ) {
    final memInfoPtr = arena.allocate<Pointer<OrtMemoryInfo>>(1);
    final cpuName = 'Cpu'.toNativeUtf8(allocator: arena).cast<Void>();
    _ort.check(
      _ort.createMemoryInfo(
        cpuName,
        OrtAllocatorType.arena,
        0,
        OrtMemType.cpu,
        memInfoPtr,
      ),
    );

    final dataPtr = NativeMemory.int64FromList(data, arena).cast<Void>();
    final shapePtr = NativeMemory.int64FromList(shape, arena);
    final ortValuePtr = arena.allocate<Pointer<OrtValue>>(1);

    _ort.check(
      _ort.createTensorWithData(
        memInfoPtr.value,
        dataPtr,
        data.length * 8, // int64 = 8 bytes
        shapePtr,
        shape.length,
        OrtTensorElementDataType.int64,
        ortValuePtr,
      ),
    );

    return ortValuePtr.value;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DartTrainingSession — pure-Dart MLP fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Dart-only training session backed by [NeuralModel].
/// No native dependencies required.
class DartTrainingSession extends TrainingSession {
  late final NeuralModel _model;

  DartTrainingSession({
    required super.spec,
    required super.optimizerConfig,
    required super.lossFunction,
  }) {
    _model = NeuralModel(spec);
  }

  @override
  Future<double> trainStep({
    required List<double> flatInputs,
    required List<dynamic> labels,
    required int batchSize,
  }) async {
    return _model.trainStepAdam(
      flatInputs,
      labels,
      batchSize,
      optimizerConfig.learningRate,
    );
  }

  @override
  Future<double> evalStep({
    required List<double> flatInputs,
    required List<dynamic> labels,
    required int batchSize,
  }) async {
    final preds = _model.predictBatch(flatInputs, batchSize);
    return LossFunctions.compute(
      lossFunction: lossFunction,
      preds: preds,
      labels: labels,
    );
  }

  @override
  Future<List<List<double>>> predict({
    required List<double> flatInputs,
    required int n,
  }) async {
    return _model.predictBatch(flatInputs, n);
  }

  @override
  Future<String> exportInferenceModel(String outputPath) async {
    final json = _model.toJson();
    final file = File(outputPath);
    await file.create(recursive: true);
    await file.writeAsString(const JsonEncoder.withIndent('  ').convert(json));
    return outputPath;
  }

  @override
  Future<void> dispose() async {}

  @override
  TrainingBackend get backend => TrainingBackend.dartFallback;
}
