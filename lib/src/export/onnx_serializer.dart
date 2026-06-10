// ignore_for_file: lines_longer_than_80_chars

import 'dart:typed_data';

// ─────────────────────────────────────────────────────────────────────────────
// OnnxSerializer — minimal ONNX protobuf builder
// ─────────────────────────────────────────────────────────────────────────────

/// Builds a minimal ONNX ModelProto FlatBuffer/Protobuf in-memory.
///
/// This is used when the Dart-fallback backend is active and we want to emit
/// a valid (if small) ONNX graph describing the trained linear/MLP model.
///
/// ONNX uses Protocol Buffers (proto3). We hand-encode only the small subset
/// we need — no external proto library required.
///
/// Produced graph:
/// ```
/// input:float[batch, featureCount]
///   → Gemm (W1, b1)
///   → Relu
///   [→ Gemm (W2, b2) → Relu]   ← optional hidden layers
///   → Gemm (Wout, bout)
///   → [Softmax]                  ← classifiers only
/// output:float[batch, outputSize]
/// ```
class OnnxSerializer {
  OnnxSerializer._();

  /// Builds the ONNX ModelProto bytes for a sequence of dense layers.
  ///
  /// [layerWeights] — list of (weights, biases) pairs per layer.
  ///   Each weights matrix is [inSize × outSize] row-major.
  /// [inputFeatures] — size of the model input vector.
  /// [outputSize] — size of the final output (num classes or 1).
  /// [isClassifier] — if true, appends a Softmax output node.
  static Uint8List build({
    required List<_LayerData> layers,
    required int inputFeatures,
    required int outputSize,
    required bool isClassifier,
    String opsetVersion = '17',
  }) {
    final buf = _ProtoWriter();

    // ModelProto fields:
    //   1: ir_version (int64) = 8
    //   8: opset_import (OpsetImportProto) — field 2: version int64
    //   7: graph (GraphProto)

    buf.writeInt64Field(1, 8); // ir_version = 8

    // opset_import { domain: "", version: 17 }
    final opset = _ProtoWriter();
    opset.writeStringField(1, '');  // domain
    opset.writeInt64Field(2, 17);   // version
    buf.writeBytesField(8, opset.bytes);

    // Build graph
    final graph = _buildGraph(
      layers: layers,
      inputFeatures: inputFeatures,
      outputSize: outputSize,
      isClassifier: isClassifier,
    );
    buf.writeBytesField(7, graph);

    // model_version
    buf.writeInt64Field(5, 1);

    // doc_string
    buf.writeStringField(12, 'offline_ml_pipeline Dart export v0.1.0');

    return buf.bytes;
  }

  static Uint8List _buildGraph({
    required List<_LayerData> layers,
    required int inputFeatures,
    required int outputSize,
    required bool isClassifier,
  }) {
    final g = _ProtoWriter();

    // Graph name (field 10)
    g.writeStringField(10, 'offline_ml_pipeline_graph');

    // ── Input (field 11) ──────────────────────────────────────────────────
    g.writeBytesField(11, _buildValueInfo('input', [0, inputFeatures]));

    // ── Output (field 12) ─────────────────────────────────────────────────
    g.writeBytesField(12, _buildValueInfo('output', [0, outputSize]));

    // ── Nodes + initializers (weights/biases) ─────────────────────────────
    String prevOutput = 'input';
    for (int li = 0; li < layers.length; li++) {
      final layer = layers[li];
      final isLast = li == layers.length - 1;
      final weightName = 'W$li';
      final biasName = 'b$li';
      final gemmOut = isLast && !isClassifier ? 'output' : 'gemm_out_$li';
      final reluOut = isLast ? 'output' : 'relu_out_$li';

      // Initializer: weight tensor [inSize, outSize]
      g.writeBytesField(
        5,
        _buildTensorProto(weightName, layer.weights,
            [layer.inSize, layer.outSize]),
      );

      // Initializer: bias tensor [outSize]
      g.writeBytesField(
        5,
        _buildTensorProto(biasName, layer.biases, [layer.outSize]),
      );

      // Gemm node: C = alpha * A * B^T + beta * bias
      // A = prevOutput [batch, inSize]
      // B = weights [inSize, outSize] → Gemm treats as [outSize, inSize] with transB=1
      final gemm = _ProtoWriter();
      gemm.writeStringField(1, 'Gemm');     // op_type
      gemm.writeStringField(7, 'Gemm_$li'); // name
      // inputs: A, B, C
      gemm.writeStringField(2, prevOutput);
      gemm.writeStringField(2, weightName);
      gemm.writeStringField(2, biasName);
      // output
      gemm.writeStringField(3, isLast && !isClassifier ? 'output' : gemmOut);
      // attribute: transB = 0 (weights already [in, out])
      gemm.writeBytesField(4, _buildAttribute('transB', 0));
      g.writeBytesField(1, gemm.bytes);

      if (!isLast) {
        // Relu activation
        final relu = _ProtoWriter();
        relu.writeStringField(1, 'Relu');
        relu.writeStringField(7, 'Relu_$li');
        relu.writeStringField(2, gemmOut);
        relu.writeStringField(3, reluOut);
        g.writeBytesField(1, relu.bytes);
        prevOutput = reluOut;
      } else if (isClassifier) {
        // Softmax on last layer
        final sm = _ProtoWriter();
        sm.writeStringField(1, 'Softmax');
        sm.writeStringField(7, 'Softmax_out');
        sm.writeStringField(2, gemmOut);
        sm.writeStringField(3, 'output');
        g.writeBytesField(1, sm.bytes);
      }
    }

    return g.bytes;
  }

  static Uint8List _buildValueInfo(String name, List<int> shape) {
    final vi = _ProtoWriter();
    vi.writeStringField(1, name);

    // type (field 2) = TypeProto
    final tp = _ProtoWriter();
    // tensor_type (field 1) = Tensor
    final tt = _ProtoWriter();
    tt.writeInt32Field(1, 1); // elem_type = FLOAT (1)

    // shape
    final shapeProto = _ProtoWriter();
    for (final dim in shape) {
      final dimProto = _ProtoWriter();
      if (dim == 0) {
        dimProto.writeStringField(2, 'batch_size'); // dim_param
      } else {
        dimProto.writeInt64Field(1, dim); // dim_value
      }
      shapeProto.writeBytesField(1, dimProto.bytes);
    }
    tt.writeBytesField(2, shapeProto.bytes);

    tp.writeBytesField(1, tt.bytes);
    vi.writeBytesField(2, tp.bytes);
    return vi.bytes;
  }

  static Uint8List _buildTensorProto(
    String name,
    List<double> data,
    List<int> dims,
  ) {
    final t = _ProtoWriter();
    t.writeStringField(8, name);          // name
    for (final d in dims) t.writeInt64Field(1, d); // dims
    t.writeInt32Field(2, 1);              // data_type = FLOAT
    // float_data (field 4) — packed floats
    for (final v in data) t.writeFloat32Field(4, v);
    return t.bytes;
  }

  static Uint8List _buildAttribute(String name, int intValue) {
    final a = _ProtoWriter();
    a.writeStringField(1, name);
    a.writeInt64Field(4, intValue); // i = int value (field 4 in AttributeProto)
    a.writeInt32Field(20, 1);       // type = INT (1)
    return a.bytes;
  }
}

// ── Helper data classes ────────────────────────────────────────────────────

class _LayerData {
  final List<double> weights; // [inSize × outSize] row-major
  final List<double> biases;  // [outSize]
  final int inSize;
  final int outSize;

  const _LayerData({
    required this.weights,
    required this.biases,
    required this.inSize,
    required this.outSize,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// _ProtoWriter — minimal protobuf binary encoder (proto3 wire format)
// ─────────────────────────────────────────────────────────────────────────────

/// Hand-written proto3 binary encoder.
/// Supports field types used by ONNX ModelProto (int32, int64, string, bytes).
class _ProtoWriter {
  final _buf = <int>[];

  Uint8List get bytes => Uint8List.fromList(_buf);

  // Wire type 0 = varint, 2 = length-delimited

  void writeInt32Field(int fieldNumber, int value) {
    _writeTag(fieldNumber, 0);
    _writeVarint(value);
  }

  void writeInt64Field(int fieldNumber, int value) {
    _writeTag(fieldNumber, 0);
    _writeVarint(value);
  }

  void writeStringField(int fieldNumber, String value) {
    final encoded = _utf8Encode(value);
    _writeTag(fieldNumber, 2);
    _writeVarint(encoded.length);
    _buf.addAll(encoded);
  }

  void writeBytesField(int fieldNumber, Uint8List data) {
    _writeTag(fieldNumber, 2);
    _writeVarint(data.length);
    _buf.addAll(data);
  }

  void writeFloat32Field(int fieldNumber, double value) {
    _writeTag(fieldNumber, 5); // wire type 5 = 32-bit
    final bd = ByteData(4)..setFloat32(0, value, Endian.little);
    _buf.addAll(bd.buffer.asUint8List());
  }

  void _writeTag(int fieldNumber, int wireType) {
    _writeVarint((fieldNumber << 3) | wireType);
  }

  void _writeVarint(int value) {
    // Handles up to 64-bit unsigned integers
    while (value > 0x7F) {
      _buf.add((value & 0x7F) | 0x80);
      value >>= 7;
    }
    _buf.add(value & 0x7F);
  }

  List<int> _utf8Encode(String s) {
    final result = <int>[];
    for (final rune in s.runes) {
      if (rune < 0x80) {
        result.add(rune);
      } else if (rune < 0x800) {
        result.add(0xC0 | (rune >> 6));
        result.add(0x80 | (rune & 0x3F));
      } else {
        result.add(0xE0 | (rune >> 12));
        result.add(0x80 | ((rune >> 6) & 0x3F));
        result.add(0x80 | (rune & 0x3F));
      }
    }
    return result;
  }
}
