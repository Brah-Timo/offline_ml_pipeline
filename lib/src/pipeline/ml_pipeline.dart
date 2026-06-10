// ignore_for_file: lines_longer_than_80_chars

import 'dart:io';
import '../data/csv_loader.dart';
import '../data/data_normalizer.dart';
import '../data/data_splitter.dart';
import '../data/feature_encoder.dart';
import '../models/base_model.dart';
import '../training/training_session.dart';
import '../training/training_loop.dart';
import '../export/model_exporter.dart';
import '../utils/progress_notifier.dart';
import 'pipeline_config.dart';
import 'pipeline_result.dart';

// ─────────────────────────────────────────────────────────────────────────────
// MlPipeline — the single entry-point class for the package
// ─────────────────────────────────────────────────────────────────────────────

/// Trains a machine-learning model on a local CSV file and exports it as a
/// `.tflite` model — entirely on-device, without any network calls.
///
/// ## Minimal example
/// ```dart
/// final pipeline = MlPipeline(
///   config: PipelineConfig(
///     csvPath: '/sdcard/Download/iris.csv',
///     targetColumn: 'species',
///     modelType: ModelType.classifier,
///   ),
/// );
///
/// // Optional: listen to training progress
/// pipeline.progressStream.listen((p) {
///   print('Epoch ${p.epoch}/${p.totalEpochs}  loss=${p.trainLoss}');
/// });
///
/// // Train — runs in a background Isolate so the UI stays responsive
/// final result = await pipeline.train();
///
/// print(result.metrics.toReport(ModelType.classifier));
/// print('Model file: ${result.tflitePath}');
/// ```
///
/// ## Training stages
/// 1. Load & parse CSV  
/// 2. Infer data schema (column types, class labels)  
/// 3. Encode categorical columns  
/// 4. Normalise numeric columns  
/// 5. Split into train / validation / test sets  
/// 6. Initialise ORT training session (or Dart fallback)  
/// 7. Run training loop with progress streaming  
/// 8. Compute final metrics on validation set  
/// 9. Export model to `.tflite`  
/// 10. Persist metadata JSON sidecar  
class MlPipeline {
  final PipelineConfig config;
  final ProgressNotifier _progressNotifier = ProgressNotifier();

  MlPipeline({required this.config});

  // ── Public interface ───────────────────────────────────────────────────

  /// Stream that emits a [TrainingProgress] event after every training epoch.
  ///
  /// Listen **before** calling [train] to receive all events.
  Stream<TrainingProgress> get progressStream => _progressNotifier.stream;

  /// Runs the full training pipeline.
  ///
  /// Returns a [PipelineResult] containing:
  /// - Path to the exported `.tflite` model.
  /// - All training metrics.
  /// - Serialised preprocessing state for inference.
  ///
  /// Runs on the calling isolate. To keep the Flutter UI thread unblocked,
  /// wrap this call with `compute()` or run it inside a `Future` on a
  /// background isolate that has been properly initialised for platform
  /// channels (see `BackgroundIsolateBinaryMessenger.ensureInitialized`).
  Future<PipelineResult> train() async {
    return _trainInternal();
  }

  // ── Internal training pipeline ─────────────────────────────────────────

  Future<PipelineResult> _trainInternal() async {
    // ── Stage 1: Load CSV ────────────────────────────────────────────────
    final loader = CsvLoader(
      config.csvPath,
      hasHeader: config.csvHasHeader,
      delimiter: config.csvDelimiter,
      encoding: config.csvEncoding,
    );
    final rawTable = await loader.load();

    // ── Stage 2: Infer schema ────────────────────────────────────────────
    final schema = rawTable.inferSchema(targetColumn: config.targetColumn);

    // ── Stage 3: Encode categorical columns ──────────────────────────────
    final encoder = FeatureEncoder(schema: schema);
    final encodedTable = encoder.fitTransform(rawTable);

    // Re-infer schema on the encoded table (column set may have changed
    // due to one-hot expansion).
    final encodedSchema = encodedTable.inferSchema(
      targetColumn: config.targetColumn,
    );

    // ── Stage 4: Normalise numeric columns ───────────────────────────────
    final normalizer = DataNormalizer(
      strategy: config.normalizationStrategy,
    );
    final normTable = normalizer.fitTransform(encodedTable, encodedSchema);

    // ── Stage 5: Split data ──────────────────────────────────────────────
    final splitter = DataSplitter(
      trainRatio: config.trainRatio,
      valRatio: config.valRatio,
      testRatio: config.testRatio,
      shuffle: config.shuffleData,
      seed: config.randomSeed,
    );
    final splits = splitter.split(normTable, encodedSchema);

    // ── Stage 6: Build ModelSpec ─────────────────────────────────────────
    final spec = ModelSpec(
      modelType: config.modelType,
      inputFeatures: encodedSchema.featureCount,
      outputSize: config.modelType.name == 'classifier'
          ? encodedSchema.outputClasses
          : 1,
      architecture: config.architecture,
    );

    // ── Stage 7: Create training session ─────────────────────────────────
    final artifactDir = await _resolveArtifactDir();

    final session = await TrainingSession.create(
      spec: spec,
      artifactDir: artifactDir,
      optimizerConfig: config.optimizerConfig,
      lossFunction: config.lossFunction,
    );

    // ── Stage 8: Training loop ────────────────────────────────────────────
    final loop = TrainingLoop(
      session: session,
      trainData: splits.train,
      valData: splits.validation,
      epochs: config.epochs,
      batchSize: config.batchSize,
      progressNotifier: _progressNotifier,
      earlyStopping: config.earlyStopping,
      initialLr: config.optimizerConfig.learningRate,
      lrSchedule: config.optimizerConfig.schedule,
    );

    final metrics = await loop.run();

    // ── Stage 9: Export model ─────────────────────────────────────────────
    final outputDir = config.outputDirectory ?? await _defaultOutputDir();
    await Directory(outputDir).create(recursive: true);

    final modelFileName =
        'offline_ml_${config.modelType.name}_${DateTime.now().millisecondsSinceEpoch}.tflite';
    final modelPath = '$outputDir/$modelFileName';

    final exporter = ModelExporter(session: session, spec: spec);
    final finalPath = await exporter.exportToTflite(
      outputPath: modelPath,
      normalizer: normalizer,
      encoder: encoder,
      embedPreprocessing: config.embedPreprocessing,
      quantizationMode: config.quantizationMode,
    );

    // ── Stage 10: Build & return result ──────────────────────────────────
    await session.dispose();
    _progressNotifier.close();

    final result = PipelineResult(
      modelPath: finalPath,
      metrics: metrics,
      schema: encodedSchema,
      normalizerState: normalizer.serialize(),
      encoderState: encoder.serialize(),
      trainingDuration: loop.duration,
      epochHistory: loop.epochHistory,
      backend: session.backend.name,
    );

    // Auto-save sidecar JSON
    await result.saveMetadata();

    return result;
  }

  // ── Helpers ────────────────────────────────────────────────────────────

  Future<String> _resolveArtifactDir() async {
    // Use the caller-supplied outputDirectory as the base when available,
    // otherwise fall back to the system temp directory. Both paths are
    // always accessible without platform-channel calls (no Flutter binding
    // required), which means this works in unit/integration tests too.
    final base = config.outputDirectory ?? Directory.systemTemp.path;
    final artifactDir = '$base/ort_training_artifacts';
    if (!Directory(artifactDir).existsSync()) {
      await Directory(artifactDir).create(recursive: true);
    }
    return artifactDir;
  }

  Future<String> _defaultOutputDir() async {
    // Prefer the caller-supplied directory; fall back to system temp.
    // path_provider's getApplicationDocumentsDirectory() requires
    // WidgetsFlutterBinding to be initialised and is therefore unavailable
    // in pure-Dart test environments. Callers running in a full Flutter app
    // can pass an explicit outputDirectory via PipelineConfig.
    return config.outputDirectory ?? Directory.systemTemp.path;
  }
}
