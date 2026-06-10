// ─────────────────────────────────────────────────────────────────────────────
// OptimizerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the gradient-descent optimiser used during training.
///
/// Pass an instance to [PipelineConfig.optimizerConfig].
///
/// Named constructors exist for the most common optimisers:
/// ```dart
/// OptimizerConfig.adam(learningRate: 0.001)
/// OptimizerConfig.sgd(learningRate: 0.01, momentum: 0.9)
/// OptimizerConfig.adaGrad(learningRate: 0.01)
/// ```
class OptimizerConfig {
  final OptimizerType type;

  /// Base learning rate.
  final double learningRate;

  // ── SGD / momentum ─────────────────────────────────────────────────────
  /// Momentum factor (only used by SGD).
  final double momentum;

  /// Nesterov momentum flag (only used by SGD).
  final bool nesterov;

  // ── Adam ───────────────────────────────────────────────────────────────
  /// First-moment exponential decay rate β₁ (Adam only).
  final double beta1;

  /// Second-moment exponential decay rate β₂ (Adam only).
  final double beta2;

  /// Numerical stability epsilon ε (Adam & AdaGrad).
  final double epsilon;

  // ── Learning-rate schedule ─────────────────────────────────────────────
  /// Optional learning-rate schedule applied every epoch.
  final LrSchedule? schedule;

  /// L2 weight-decay coefficient (applied to all weight matrices).
  final double weightDecay;

  const OptimizerConfig._({
    required this.type,
    required this.learningRate,
    this.momentum = 0.0,
    this.nesterov = false,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
    this.schedule,
    this.weightDecay = 1e-4,
  });

  // ── Named constructors ─────────────────────────────────────────────────

  /// **Adam** optimiser (Adaptive Moment Estimation).
  ///
  /// Default choice for most problems.
  const factory OptimizerConfig.adam({
    double learningRate,
    double beta1,
    double beta2,
    double epsilon,
    LrSchedule? schedule,
    double weightDecay,
  }) = _AdamConfig;

  /// **SGD** with optional Nesterov momentum.
  const factory OptimizerConfig.sgd({
    double learningRate,
    double momentum,
    bool nesterov,
    LrSchedule? schedule,
    double weightDecay,
  }) = _SgdConfig;

  /// **AdaGrad** — adapts the learning rate per parameter.
  const factory OptimizerConfig.adaGrad({
    double learningRate,
    double epsilon,
    double weightDecay,
  }) = _AdaGradConfig;

  // ── Serialisation ──────────────────────────────────────────────────────

  Map<String, dynamic> toJson() => {
        'type': type.name,
        'learningRate': learningRate,
        'momentum': momentum,
        'nesterov': nesterov,
        'beta1': beta1,
        'beta2': beta2,
        'epsilon': epsilon,
        'weightDecay': weightDecay,
        'schedule': schedule?.toJson(),
      };

  factory OptimizerConfig.fromJson(Map<String, dynamic> j) {
    final type = OptimizerType.values.firstWhere(
      (e) => e.name == j['type'],
    );
    switch (type) {
      case OptimizerType.adam:
        return OptimizerConfig.adam(
          learningRate: (j['learningRate'] as num).toDouble(),
          beta1: (j['beta1'] as num).toDouble(),
          beta2: (j['beta2'] as num).toDouble(),
          epsilon: (j['epsilon'] as num).toDouble(),
          weightDecay: (j['weightDecay'] as num).toDouble(),
        );
      case OptimizerType.sgd:
        return OptimizerConfig.sgd(
          learningRate: (j['learningRate'] as num).toDouble(),
          momentum: (j['momentum'] as num).toDouble(),
          nesterov: j['nesterov'] as bool,
          weightDecay: (j['weightDecay'] as num).toDouble(),
        );
      case OptimizerType.adaGrad:
        return OptimizerConfig.adaGrad(
          learningRate: (j['learningRate'] as num).toDouble(),
          epsilon: (j['epsilon'] as num).toDouble(),
          weightDecay: (j['weightDecay'] as num).toDouble(),
        );
    }
  }
}

// ── Concrete named-constructor classes ────────────────────────────────────

class _AdamConfig extends OptimizerConfig {
  const _AdamConfig({
    double learningRate = 0.001,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double epsilon = 1e-8,
    LrSchedule? schedule,
    double weightDecay = 1e-4,
  }) : super._(
          type: OptimizerType.adam,
          learningRate: learningRate,
          beta1: beta1,
          beta2: beta2,
          epsilon: epsilon,
          schedule: schedule,
          weightDecay: weightDecay,
        );
}

class _SgdConfig extends OptimizerConfig {
  const _SgdConfig({
    double learningRate = 0.01,
    double momentum = 0.9,
    bool nesterov = false,
    LrSchedule? schedule,
    double weightDecay = 1e-4,
  }) : super._(
          type: OptimizerType.sgd,
          learningRate: learningRate,
          momentum: momentum,
          nesterov: nesterov,
          schedule: schedule,
          weightDecay: weightDecay,
        );
}

class _AdaGradConfig extends OptimizerConfig {
  const _AdaGradConfig({
    double learningRate = 0.01,
    double epsilon = 1e-8,
    double weightDecay = 1e-4,
  }) : super._(
          type: OptimizerType.adaGrad,
          learningRate: learningRate,
          epsilon: epsilon,
          weightDecay: weightDecay,
        );
}

// ─────────────────────────────────────────────────────────────────────────────
// OptimizerType
// ─────────────────────────────────────────────────────────────────────────────

enum OptimizerType { adam, sgd, adaGrad }

// ─────────────────────────────────────────────────────────────────────────────
// LrSchedule
// ─────────────────────────────────────────────────────────────────────────────

/// Optional learning-rate decay schedule.
class LrSchedule {
  final LrScheduleType type;

  /// Multiplicative decay factor (step / exponential decay).
  final double factor;

  /// Epoch interval for step-decay.
  final int stepSize;

  /// Minimum allowed learning rate.
  final double minLr;

  const LrSchedule({
    required this.type,
    this.factor = 0.1,
    this.stepSize = 10,
    this.minLr = 1e-6,
  });

  /// Returns the learning rate for [epoch] given an [initialLr].
  double compute(double initialLr, int epoch) {
    switch (type) {
      case LrScheduleType.stepDecay:
        final steps = epoch ~/ stepSize;
        return (initialLr * _pow(factor, steps)).clamp(minLr, double.infinity);
      case LrScheduleType.exponentialDecay:
        return (initialLr * _pow(factor, epoch)).clamp(minLr, double.infinity);
      case LrScheduleType.cosineAnnealing:
        // lr = minLr + 0.5*(initialLr - minLr)*(1 + cos(π * epoch / T))
        // We use 100 as default period T.
        const t = 100;
        return minLr +
            0.5 *
                (initialLr - minLr) *
                (1 + _cos(3.14159265 * epoch / t));
      case LrScheduleType.warmupCosine:
        if (epoch < stepSize) {
          // Linear warmup
          return initialLr * epoch / stepSize;
        }
        return minLr +
            0.5 *
                (initialLr - minLr) *
                (1 + _cos(3.14159265 * (epoch - stepSize) / 100));
    }
  }

  Map<String, dynamic> toJson() => {
        'type': type.name,
        'factor': factor,
        'stepSize': stepSize,
        'minLr': minLr,
      };

  factory LrSchedule.fromJson(Map<String, dynamic> j) => LrSchedule(
        type: LrScheduleType.values.firstWhere((e) => e.name == j['type']),
        factor: (j['factor'] as num).toDouble(),
        stepSize: j['stepSize'] as int,
        minLr: (j['minLr'] as num).toDouble(),
      );
}

enum LrScheduleType {
  stepDecay,
  exponentialDecay,
  cosineAnnealing,
  warmupCosine,
}

// ── Pure-Dart math helpers ─────────────────────────────────────────────────
double _pow(double base, int exp) {
  double result = 1.0;
  for (int i = 0; i < exp; i++) {
    result *= base;
  }
  return result;
}

double _cos(double x) {
  double r = 1, term = 1;
  for (int k = 1; k <= 10; k++) {
    term *= -x * x / ((2 * k - 1) * (2 * k));
    r += term;
  }
  return r;
}
