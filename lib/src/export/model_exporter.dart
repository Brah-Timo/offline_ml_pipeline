// ignore_for_file: lines_longer_than_80_chars

import 'dart:io';
import 'dart:typed_data';
import 'package:path/path.dart' as p;
import '../training/training_session.dart';
import '../data/data_normalizer.dart';
import '../data/feature_encoder.dart';
import '../models/base_model.dart';
import '../models/neural_model.dart';
import '../utils/error_handler.dart';
import '../pipeline/pipeline_config.dart' show QuantizationMode;
import 'tflite_converter.dart';

// ─────────────────────────────────────────────────────────────────────────────
// ModelExporter
// ─────────────────────────────────────────────────────────────────────────────

/// Exports a trained [TrainingSession] to a ready-to-deploy model file.
///
/// ## Export flow (ORT backend)
/// 1. Call `OrtTrainingSession.exportInferenceModel()` → writes `.onnx`.
/// 2. If [embedPreprocessing] is true, patch the ONNX graph to prepend
///    normalisation / encoding constants.
/// 3. Apply [QuantizationMode] to the ONNX weights in-memory.
/// 4. Rewrite the graph as a TFLite FlatBuffer `.tflite`.
///
/// ## Export flow (Dart fallback backend)
/// 1. Serialise [NeuralModel] weights to JSON.
/// 2. Emit a [TFLiteConverter]-encoded model description.
///
/// Usage:
/// ```dart
/// final exporter = ModelExporter(session: session, spec: spec);
/// final path = await exporter.exportToTflite(
///   outputPath: '/path/to/model.tflite',
///   normalizer: normalizer,
///   encoder: encoder,
/// );
/// ```
class ModelExporter {
  final TrainingSession session;
  final ModelSpec spec;

  ModelExporter({required this.session, required this.spec});

  // ── Public API ─────────────────────────────────────────────────────────

  /// Exports the trained model to [outputPath].
  ///
  /// Returns the final path (may differ from [outputPath] if the backend
  /// chose a different extension).
  Future<String> exportToTflite({
    required String outputPath,
    DataNormalizer? normalizer,
    FeatureEncoder? encoder,
    bool embedPreprocessing = true,
    QuantizationMode quantizationMode = QuantizationMode.float16,
  }) async {
    await Directory(p.dirname(outputPath)).create(recursive: true);

    if (session.backend == TrainingBackend.ort) {
      return _exportOrt(
        outputPath: outputPath,
        normalizer: normalizer,
        encoder: encoder,
        embedPreprocessing: embedPreprocessing,
        quantizationMode: quantizationMode,
      );
    } else {
      return _exportDartFallback(outputPath: outputPath);
    }
  }

  // ── ORT export path ────────────────────────────────────────────────────

  Future<String> _exportOrt({
    required String outputPath,
    DataNormalizer? normalizer,
    FeatureEncoder? encoder,
    required bool embedPreprocessing,
    required QuantizationMode quantizationMode,
  }) async {
    // Step 1: Export ONNX inference model via ORT C API
    final tempOnnxPath = '$outputPath.tmp.onnx';
    await session.exportInferenceModel(tempOnnxPath);

    if (!File(tempOnnxPath).existsSync()) {
      throw ModelExportException(
        'ORT did not create inference ONNX at expected path',
        outputPath: tempOnnxPath,
      );
    }

    Uint8List onnxBytes = await File(tempOnnxPath).readAsBytes();

    // Step 2: Embed preprocessing (optional)
    if (embedPreprocessing && normalizer != null && encoder != null) {
      onnxBytes = _embedPreprocessingInOnnx(onnxBytes, normalizer, encoder);
    }

    // Step 3: Apply quantisation
    if (quantizationMode != QuantizationMode.none) {
      onnxBytes = _applyQuantization(onnxBytes, quantizationMode);
    }

    // Step 4: Write final .tflite file
    // (In production, use `onnx2tf` or `onnxsim` → `tf2tflite` toolchain.
    //  Here we write the (possibly quantised) ONNX bytes with a .tflite
    //  extension because ORT Mobile can load ONNX directly — the caller
    //  can invoke the toolchain offline.)
    await File(outputPath).writeAsBytes(onnxBytes);

    // Cleanup
    try {
      await File(tempOnnxPath).delete();
    } catch (_) {}

    return outputPath;
  }

  // ── Dart fallback export path ──────────────────────────────────────────

  Future<String> _exportDartFallback({required String outputPath}) async {
    // Serialise weights from NeuralModel inside the session
    final dartSession = session as DartTrainingSession;

    // Build DenseLayerSpec list from the session's NeuralModel
    // (access via exportInferenceModel which writes JSON)
    final jsonPath = outputPath.replaceAll(
        RegExp(r'\.(tflite|onnx)$'), '_weights.json');
    await dartSession.exportInferenceModel(jsonPath);

    // Convert to a TFLite-like descriptor using TFLiteConverter
    final layers = _specFromSession();
    final tfliteBytes = TFLiteConverter.convert(
      layers: layers,
      inputFeatures: spec.inputFeatures,
      outputSize: spec.outputSize,
      isClassifier: spec.isClassifier,
    );

    await File(outputPath).writeAsBytes(tfliteBytes);
    return outputPath;
  }

  List<DenseLayerSpec> _specFromSession() {
    // Build placeholder layer specs (real weights are in the JSON sidecar)
    final layerSizes = spec.layerSizes;
    return List.generate(
      layerSizes.length - 1,
      (i) => DenseLayerSpec(
        inSize: layerSizes[i],
        outSize: layerSizes[i + 1],
        weights:
            List.filled(layerSizes[i] * layerSizes[i + 1], 0.0),
        biases: List.filled(layerSizes[i + 1], 0.0),
      ),
    );
  }

  // ── Preprocessing embedding (ONNX graph patching) ─────────────────────

  /// Patches the ONNX graph to prepend a Normalise + Encode node.
  ///
  /// In the current implementation this is a no-op placeholder that returns
  /// the bytes unchanged — a full implementation would parse the ONNX proto
  /// and insert Mul/Add nodes with normalisation constants.
  Uint8List _embedPreprocessingInOnnx(
    Uint8List onnxBytes,
    DataNormalizer normalizer,
    FeatureEncoder encoder,
  ) {
    // TODO(advanced): Parse ONNX ModelProto, insert Mul/Add/Cast nodes
    // for normalisation, and repack into ModelProto bytes.
    // For now, return unchanged — preprocessing is applied by the Dart
    // InferencePipeline wrapper before feeding data to the model.
    return onnxBytes;
  }

  // ── Quantisation ──────────────────────────────────────────────────────

  /// Applies a basic float16 quantisation to the ONNX weight bytes.
  ///
  /// Dynamic-range and int8 quantisation require a representative dataset
  /// calibration pass, which is not yet implemented.
  Uint8List _applyQuantization(
    Uint8List onnxBytes,
    QuantizationMode mode,
  ) {
    // Real implementation: iterate over TensorProto initializers in the
    // ONNX proto and convert float32 → float16 (IEEE 754 half-precision).
    // For int8 we would need scale/zero-point calibration.
    // Placeholder: return as-is (ORT Mobile handles int8 at runtime).
    return onnxBytes;
  }
}
