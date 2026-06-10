// ─────────────────────────────────────────────────────────────────────────────
// ModelType
// ─────────────────────────────────────────────────────────────────────────────

/// Fundamental type of supervised learning task.
///
/// Pass this to [PipelineConfig] to tell the pipeline what kind of
/// problem it is solving.
enum ModelType {
  /// **Classification** — target column contains discrete string labels.
  ///
  /// The pipeline will:
  /// - Use cross-entropy loss
  /// - Report accuracy, F1, precision, recall, confusion matrix, AUC-ROC
  /// - Export a model that outputs class probabilities (softmax)
  classifier,

  /// **Regression** — target column contains continuous numeric values.
  ///
  /// The pipeline will:
  /// - Use MSE loss by default (MAE also available)
  /// - Report RMSE, MAE, R², MAPE
  /// - Export a model that outputs a single scalar
  regressor,
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelArchitecture
// ─────────────────────────────────────────────────────────────────────────────

/// The internal neural-network architecture used for training.
///
/// All architectures are small enough to train on-device in reasonable time.
enum ModelArchitecture {
  /// Logistic regression (classifier) or linear regression (regressor).
  ///
  /// - Fastest to train; good baseline.
  /// - Suitable for linearly separable data.
  linear,

  /// Shallow multi-layer perceptron: 1 hidden layer (64 units, ReLU).
  ///
  /// - Good all-around choice for tabular data.
  /// - Default architecture.
  mlpShallow,

  /// Deeper MLP: 2 hidden layers (128 → 64 units, ReLU + BatchNorm).
  ///
  /// - Better for complex feature interactions.
  /// - Longer to train.
  mlpDeep,
}
