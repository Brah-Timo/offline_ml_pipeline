## 0.1.0

* Initial release.
* Full on-device training pipeline: CSV → trained model → `.tflite` export.
* ONNX Runtime On-Device Training backend via `dart:ffi`.
* Pure-Dart MLP fallback (no native dependencies required).
* Classification and regression support.
* MinMax / Z-Score normalisation.
* Ordinal + one-hot categorical encoding.
* Adam / SGD / AdaGrad optimisers.
* Early stopping.
* Progress streaming via `MlPipeline.progressStream`.
* Metrics: accuracy, F1, precision, recall, confusion matrix (classifier);
  RMSE, MAE, R², MAPE (regressor).
* Android (arm64-v8a, armeabi-v7a, x86_64) + iOS support.
* Example Flutter app with live epoch chart.
