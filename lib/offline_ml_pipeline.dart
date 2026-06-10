/// offline_ml_pipeline
///
/// Train + export ML models entirely on-device.
/// No server, no internet — complete data privacy.
///
/// Quick-start:
/// ```dart
/// final pipeline = MlPipeline(
///   config: PipelineConfig(
///     csvPath: '/sdcard/data.csv',
///     targetColumn: 'label',
///     modelType: ModelType.classifier,
///   ),
/// );
///
/// pipeline.progressStream.listen((p) =>
///     print('Epoch ${p.epoch}/${p.totalEpochs} — loss: ${p.trainLoss}'));
///
/// final result = await pipeline.train();
/// print('Accuracy : ${result.metrics.accuracy}');
/// print('TFLite   : ${result.tflitePath}');
/// ```
library offline_ml_pipeline;

// ── Public pipeline API ───────────────────────────────────────────────────
export 'src/pipeline/ml_pipeline.dart';
export 'src/pipeline/pipeline_config.dart';
export 'src/pipeline/pipeline_result.dart';

// ── Model types & enumerations ────────────────────────────────────────────
export 'src/models/model_type.dart';

// ── Training configuration ────────────────────────────────────────────────
export 'src/training/optimizer_config.dart';
export 'src/training/loss_functions.dart';
export 'src/training/training_loop.dart' show EarlyStopping, TrainingLoop;
export 'src/training/training_metrics.dart';
export 'src/training/training_session.dart' show TrainingSession, DartTrainingSession, TrainingBackend;

// ── Data layer (for advanced users) ──────────────────────────────────────
export 'src/data/csv_loader.dart'
    show CsvLoader, RawDataTable, CsvFileNotFoundException, EmptyCsvException;
export 'src/data/data_schema.dart';
export 'src/data/data_normalizer.dart' show DataNormalizer, NormalizationStrategy;
export 'src/data/data_splitter.dart' show DataSplitter, DataSplits;
export 'src/data/feature_encoder.dart' show FeatureEncoder;

// ── Errors & exceptions ───────────────────────────────────────────────────
export 'src/utils/error_handler.dart';

// ── Progress types ────────────────────────────────────────────────────────
export 'src/utils/progress_notifier.dart' show TrainingProgress;
