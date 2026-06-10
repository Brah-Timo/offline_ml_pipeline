// ignore_for_file: lines_longer_than_80_chars

import 'dart:math' as math;
import 'base_model.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Decision-Tree Node
// ─────────────────────────────────────────────────────────────────────────────

class _TreeNode {
  // Internal (split) node fields
  int featureIndex;
  double threshold;
  _TreeNode? left;
  _TreeNode? right;

  // Leaf node field
  // For classifier: index of the majority class.
  // For regressor: mean target value.
  double? leafValue;

  bool get isLeaf => leafValue != null;

  _TreeNode({
    this.featureIndex = 0,
    this.threshold = 0.0,
    this.left,
    this.right,
    this.leafValue,
  });

  Map<String, dynamic> toJson() => {
        'fi': featureIndex,
        'th': threshold,
        'lv': leafValue,
        'l': left?.toJson(),
        'r': right?.toJson(),
      };

  factory _TreeNode.fromJson(Map<String, dynamic> j) {
    return _TreeNode(
      featureIndex: j['fi'] as int,
      threshold: (j['th'] as num).toDouble(),
      leafValue:
          j['lv'] == null ? null : (j['lv'] as num).toDouble(),
      left: j['l'] == null
          ? null
          : _TreeNode.fromJson(j['l'] as Map<String, dynamic>),
      right: j['r'] == null
          ? null
          : _TreeNode.fromJson(j['r'] as Map<String, dynamic>),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// TreeModel — CART Decision Tree (pure Dart)
// ─────────────────────────────────────────────────────────────────────────────

/// Pure-Dart CART (Classification and Regression Trees) decision tree.
///
/// Supports both classification (Gini impurity splitting) and
/// regression (MSE splitting) tasks.
///
/// Parameters:
/// - [maxDepth]: Maximum tree depth (default 8).
/// - [minSamplesSplit]: Minimum samples required to split a node (default 4).
/// - [minSamplesLeaf]: Minimum samples in a leaf (default 2).
class TreeModel {
  final ModelSpec spec;
  final int maxDepth;
  final int minSamplesSplit;
  final int minSamplesLeaf;

  _TreeNode? _root;
  bool _trained = false;

  TreeModel(
    this.spec, {
    this.maxDepth = 8,
    this.minSamplesSplit = 4,
    this.minSamplesLeaf = 2,
  });

  // ── Training ───────────────────────────────────────────────────────────

  /// Trains the tree on the entire provided dataset (no mini-batch).
  void fit(List<double> flatFeatures, List<dynamic> labels, int n) {
    final indices = List<int>.generate(n, (i) => i);
    _root = _buildTree(flatFeatures, labels, indices, depth: 0);
    _trained = true;
  }

  // ── Prediction ─────────────────────────────────────────────────────────

  double predictOne(List<double> features) {
    if (!_trained || _root == null) return 0.0;
    return _traverse(_root!, features);
  }

  List<double> predictBatch(List<double> flat, int n) {
    final f = spec.inputFeatures;
    return List.generate(n, (i) {
      return predictOne(flat.sublist(i * f, (i + 1) * f));
    });
  }

  // ── Internal CART construction ─────────────────────────────────────────

  _TreeNode _buildTree(
    List<double> flat,
    List<dynamic> labels,
    List<int> indices,
    {required int depth}
  ) {
    // Stopping criteria
    if (depth >= maxDepth ||
        indices.length < minSamplesSplit ||
        _isPure(labels, indices)) {
      return _TreeNode(leafValue: _leafValue(labels, indices));
    }

    final best = _bestSplit(flat, labels, indices);
    if (best == null) {
      return _TreeNode(leafValue: _leafValue(labels, indices));
    }

    final leftIdx = <int>[];
    final rightIdx = <int>[];
    final f = spec.inputFeatures;

    for (final i in indices) {
      if (flat[i * f + best.featureIndex] <= best.threshold) {
        leftIdx.add(i);
      } else {
        rightIdx.add(i);
      }
    }

    if (leftIdx.length < minSamplesLeaf ||
        rightIdx.length < minSamplesLeaf) {
      return _TreeNode(leafValue: _leafValue(labels, indices));
    }

    final node = _TreeNode(
      featureIndex: best.featureIndex,
      threshold: best.threshold,
    );

    node.left = _buildTree(flat, labels, leftIdx, depth: depth + 1);
    node.right = _buildTree(flat, labels, rightIdx, depth: depth + 1);
    return node;
  }

  _SplitResult? _bestSplit(
    List<double> flat,
    List<dynamic> labels,
    List<int> indices,
  ) {
    final f = spec.inputFeatures;
    double bestScore = double.infinity;
    _SplitResult? best;

    for (int fi = 0; fi < f; fi++) {
      // Collect unique threshold candidates
      final vals = indices.map((i) => flat[i * f + fi]).toSet().toList()..sort();

      for (int k = 0; k < vals.length - 1; k++) {
        final thresh = (vals[k] + vals[k + 1]) / 2;

        final leftIdx = indices.where((i) => flat[i * f + fi] <= thresh).toList();
        final rightIdx = indices.where((i) => flat[i * f + fi] > thresh).toList();

        if (leftIdx.isEmpty || rightIdx.isEmpty) continue;

        final score = spec.isClassifier
            ? _weightedGini(labels, leftIdx, rightIdx)
            : _weightedMse(labels, leftIdx, rightIdx);

        if (score < bestScore) {
          bestScore = score;
          best = _SplitResult(featureIndex: fi, threshold: thresh);
        }
      }
    }

    return best;
  }

  // ── Impurity helpers ───────────────────────────────────────────────────

  double _weightedGini(
    List<dynamic> labels,
    List<int> left,
    List<int> right,
  ) {
    final n = left.length + right.length;
    return (left.length / n) * _gini(labels, left) +
        (right.length / n) * _gini(labels, right);
  }

  double _gini(List<dynamic> labels, List<int> indices) {
    final counts = <dynamic, int>{};
    for (final i in indices) counts[labels[i]] = (counts[labels[i]] ?? 0) + 1;
    final n = indices.length.toDouble();
    double impurity = 1.0;
    for (final c in counts.values) {
      impurity -= math.pow(c / n, 2);
    }
    return impurity;
  }

  double _weightedMse(
    List<dynamic> labels,
    List<int> left,
    List<int> right,
  ) {
    final n = left.length + right.length;
    return (left.length / n) * _mse(labels, left) +
        (right.length / n) * _mse(labels, right);
  }

  double _mse(List<dynamic> labels, List<int> indices) {
    final vals =
        indices.map((i) => (labels[i] as num).toDouble()).toList();
    final mean = vals.reduce((a, b) => a + b) / vals.length;
    return vals.map((v) => math.pow(v - mean, 2).toDouble()).reduce((a, b) => a + b) /
        vals.length;
  }

  bool _isPure(List<dynamic> labels, List<int> indices) {
    if (indices.isEmpty) return true;
    final first = labels[indices[0]];
    return indices.every((i) => labels[i] == first);
  }

  double _leafValue(List<dynamic> labels, List<int> indices) {
    if (indices.isEmpty) return 0.0;
    if (spec.isClassifier) {
      final counts = <dynamic, int>{};
      for (final i in indices) counts[labels[i]] = (counts[labels[i]] ?? 0) + 1;
      return (counts.entries.reduce((a, b) => a.value > b.value ? a : b).key
              as num)
          .toDouble();
    } else {
      final vals = indices.map((i) => (labels[i] as num).toDouble());
      return vals.reduce((a, b) => a + b) / vals.length;
    }
  }

  double _traverse(_TreeNode node, List<double> features) {
    if (node.isLeaf) return node.leafValue!;
    if (features[node.featureIndex] <= node.threshold) {
      return _traverse(node.left!, features);
    }
    return _traverse(node.right!, features);
  }

  // ── Serialisation ──────────────────────────────────────────────────────

  Map<String, dynamic> toJson() => {
        'spec': spec.toJson(),
        'maxDepth': maxDepth,
        'minSamplesSplit': minSamplesSplit,
        'minSamplesLeaf': minSamplesLeaf,
        'root': _root?.toJson(),
      };

  factory TreeModel.fromJson(Map<String, dynamic> j) {
    final m = TreeModel(
      ModelSpec.fromJson(j['spec'] as Map<String, dynamic>),
      maxDepth: j['maxDepth'] as int,
      minSamplesSplit: j['minSamplesSplit'] as int,
      minSamplesLeaf: j['minSamplesLeaf'] as int,
    );
    if (j['root'] != null) {
      m._root = _TreeNode.fromJson(j['root'] as Map<String, dynamic>);
      m._trained = true;
    }
    return m;
  }
}

class _SplitResult {
  final int featureIndex;
  final double threshold;
  const _SplitResult({required this.featureIndex, required this.threshold});
}
