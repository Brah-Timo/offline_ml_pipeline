// ignore_for_file: lines_longer_than_80_chars

import '../models/model_type.dart';
import '../data/data_normalizer.dart';
import '../training/optimizer_config.dart';
import '../training/loss_functions.dart';
import '../training/training_loop.dart';

// ─────────────────────────────────────────────────────────────────────────────
// QuantizationMode
// ─────────────────────────────────────────────────────────────────────────────

/// Post-training quantisation applied when exporting to TFLite.
enum QuantizationMode {
  /// No quantisation — full float32 weights.
  none,

  /// Float16 weights — roughly 2× smaller, minimal accuracy loss.
  float16,

  /// Dynamic-range int8 quantisation — ~4× smaller, fastest on ARM.
  int8DynamicRange,

  /// Full-integer int8 quantisation — requires representative dataset.
  int8Full,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Complete configuration for an [MlPipeline] training run.
///
/// All fields have sensible defaults; only [csvPath], [targetColumn],
/// and [modelType] are required.
///
/// ## Minimal usage
/// ```dart
/// PipelineConfig(
///   csvPath: '/sdcard/data.csv',
///   targetColumn: 'label',
///   modelType: ModelType.classifier,
/// )
/// ```
///
/// ## Factory presets
/// ```dart
/// PipelineConfig.healthClassifier(csvPath: '...', targetColumn: 'diagnosis')
/// PipelineConfig.financeRegressor(csvPath: '...', targetColumn: 'close_price')
/// ```
class PipelineConfig {
  // ── Data ──────────────────────────────────────────────────────────────

  /// Absolute path to the CSV file on the device.
  final String csvPath;

  /// Name of the column to predict.
  final String targetColumn;

  /// Column delimiter — default `,`.
  final String csvDelimiter;

  /// Whether the first CSV row is a header — default `true`.
  final bool csvHasHeader;

  /// File encoding — `'utf-8'` or `'latin1'`.
  final String csvEncoding;

  // ── Task ──────────────────────────────────────────────────────────────

  /// Classification or regression.
  final ModelType modelType;

  /// Internal model architecture.
  final ModelArchitecture architecture;

  // ── Data splitting ────────────────────────────────────────────────────

  final double trainRatio;   // default 0.8
  final double valRatio;     // default 0.1
  final double testRatio;    // default 0.1
  final bool shuffleData;
  final int? randomSeed;

  // ── Normalisation ─────────────────────────────────────────────────────

  final NormalizationStrategy normalizationStrategy;

  // ── Training ──────────────────────────────────────────────────────────

  final int epochs;
  final int batchSize;
  final OptimizerConfig optimizerConfig;
  final LossFunction lossFunction;
  final EarlyStopping? earlyStopping;

  // ── Export ────────────────────────────────────────────────────────────

  /// Embed normalisation + encoding constants as a pre-processing graph
  /// inside the exported model so it is fully self-contained.
  final bool embedPreprocessing;

  /// Quantisation mode for the .tflite export.
  final QuantizationMode quantizationMode;

  /// Custom output directory; if null, uses `getApplicationDocumentsDirectory`.
  final String? outputDirectory;

  const PipelineConfig({
    required this.csvPath,
    required this.targetColumn,
    required this.modelType,
    this.csvDelimiter = ',',
    this.csvHasHeader = true,
    this.csvEncoding = 'utf-8',
    this.architecture = ModelArchitecture.mlpShallow,
    this.trainRatio = 0.8,
    this.valRatio = 0.1,
    this.testRatio = 0.1,
    this.shuffleData = true,
    this.randomSeed,
    this.normalizationStrategy = NormalizationStrategy.minMax,
    this.epochs = 100,
    this.batchSize = 32,
    this.optimizerConfig = const OptimizerConfig.adam(learningRate: 0.001),
    this.lossFunction = LossFunction.crossEntropy,
    this.earlyStopping,
    this.embedPreprocessing = true,
    this.quantizationMode = QuantizationMode.float16,
    this.outputDirectory,
  }) : assert(
          trainRatio + valRatio + testRatio - 1.0 < 0.01 &&
              trainRatio + valRatio + testRatio - 1.0 > -0.01,
          'trainRatio + valRatio + testRatio must equal 1.0',
        );

  // ── Named presets ──────────────────────────────────────────────────────

  /// Preset optimised for medical / health classification datasets.
  ///
  /// - Z-score normalisation (more robust to outliers)
  /// - 150 epochs with early stopping (patience 15)
  /// - int8 quantisation (smallest on-device footprint)
  factory PipelineConfig.healthClassifier({
    required String csvPath,
    required String targetColumn,
    String? outputDirectory,
  }) =>
      PipelineConfig(
        csvPath: csvPath,
        targetColumn: targetColumn,
        modelType: ModelType.classifier,
        normalizationStrategy: NormalizationStrategy.zScore,
        epochs: 150,
        batchSize: 16,
        optimizerConfig: const OptimizerConfig.adam(learningRate: 5e-4),
        lossFunction: LossFunction.crossEntropy,
        earlyStopping: const EarlyStopping(patience: 15, minDelta: 0.001),
        quantizationMode: QuantizationMode.int8DynamicRange,
        outputDirectory: outputDirectory,
      );

  /// Preset optimised for financial / time-series regression.
  factory PipelineConfig.financeRegressor({
    required String csvPath,
    required String targetColumn,
    String? outputDirectory,
  }) =>
      PipelineConfig(
        csvPath: csvPath,
        targetColumn: targetColumn,
        modelType: ModelType.regressor,
        normalizationStrategy: NormalizationStrategy.minMax,
        epochs: 200,
        batchSize: 64,
        optimizerConfig: const OptimizerConfig.adam(learningRate: 0.001),
        lossFunction: LossFunction.mse,
        earlyStopping: const EarlyStopping(patience: 20, minDelta: 1e-5),
        quantizationMode: QuantizationMode.float16,
        outputDirectory: outputDirectory,
      );

  /// Preset for generic classification with balanced defaults.
  factory PipelineConfig.generalClassifier({
    required String csvPath,
    required String targetColumn,
    String? outputDirectory,
  }) =>
      PipelineConfig(
        csvPath: csvPath,
        targetColumn: targetColumn,
        modelType: ModelType.classifier,
        outputDirectory: outputDirectory,
      );

  // ── Serialisation ──────────────────────────────────────────────────────

  Map<String, dynamic> toJson() => {
        'csvPath': csvPath,
        'targetColumn': targetColumn,
        'modelType': modelType.name,
        'architecture': architecture.name,
        'trainRatio': trainRatio,
        'valRatio': valRatio,
        'testRatio': testRatio,
        'shuffleData': shuffleData,
        'randomSeed': randomSeed,
        'normalizationStrategy': normalizationStrategy.name,
        'epochs': epochs,
        'batchSize': batchSize,
        'optimizerConfig': optimizerConfig.toJson(),
        'lossFunction': lossFunction.name,
        'quantizationMode': quantizationMode.name,
        'embedPreprocessing': embedPreprocessing,
      };
}
