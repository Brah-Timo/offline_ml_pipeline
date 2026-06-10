import 'package:flutter/material.dart';
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';

/// A simple line chart showing train vs. validation loss per epoch.
/// Built with CustomPainter — no external chart library needed.
class EpochChart extends StatelessWidget {
  final List<EpochRecord> history;

  const EpochChart({super.key, required this.history});

  @override
  Widget build(BuildContext context) {
    if (history.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Loss Curve',
          style: Theme.of(context).textTheme.titleSmall,
        ),
        const SizedBox(height: 8),
        SizedBox(
          height: 180,
          child: CustomPaint(
            painter: _LossCurvePainter(
              history: history,
              trainColor: Theme.of(context).colorScheme.primary,
              valColor: Theme.of(context).colorScheme.error,
            ),
            child: const SizedBox.expand(),
          ),
        ),
        const SizedBox(height: 4),
        Row(
          children: [
            _LegendDot(
              color: Theme.of(context).colorScheme.primary,
              label: 'Train loss',
            ),
            const SizedBox(width: 16),
            _LegendDot(
              color: Theme.of(context).colorScheme.error,
              label: 'Val loss',
            ),
          ],
        ),
      ],
    );
  }
}

class _LossPoint {
  final int epoch;
  final double trainLoss;
  final double valLoss;
  const _LossPoint(this.epoch, this.trainLoss, this.valLoss);
}

class _LossCurvePainter extends CustomPainter {
  final List<EpochRecord> history;
  final Color trainColor;
  final Color valColor;

  _LossCurvePainter({
    required this.history,
    required this.trainColor,
    required this.valColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (history.isEmpty) return;

    final points = history
        .map((r) => _LossPoint(r.epoch, r.trainLoss, r.valLoss))
        .toList();

    final allLosses = points
        .expand((p) => [p.trainLoss, if (p.valLoss >= 0) p.valLoss])
        .toList();

    final maxLoss = allLosses.reduce((a, b) => a > b ? a : b);
    final minLoss = allLosses.reduce((a, b) => a < b ? a : b);
    final range = (maxLoss - minLoss).clamp(1e-6, double.infinity);

    double px(int epoch) =>
        (epoch - 1) / (points.length - 1).clamp(1, double.infinity) *
        size.width;

    double py(double loss) =>
        size.height - (loss - minLoss) / range * size.height * 0.9 - 4;

    void drawLine(List<Offset> pts, Color color) {
      final paint = Paint()
        ..color = color
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round;

      final path = Path();
      if (pts.isEmpty) return;
      path.moveTo(pts[0].dx, pts[0].dy);
      for (int i = 1; i < pts.length; i++) {
        path.lineTo(pts[i].dx, pts[i].dy);
      }
      canvas.drawPath(path, paint);
    }

    // Train loss line
    drawLine(
      points.map((p) => Offset(px(p.epoch), py(p.trainLoss))).toList(),
      trainColor,
    );

    // Val loss line (if available)
    final valPts = points
        .where((p) => p.valLoss >= 0)
        .map((p) => Offset(px(p.epoch), py(p.valLoss)))
        .toList();
    if (valPts.isNotEmpty) drawLine(valPts, valColor);

    // Axis lines
    final axisPaint = Paint()
      ..color = Colors.grey.withOpacity(0.3)
      ..strokeWidth = 1;
    canvas.drawLine(Offset(0, size.height), Offset(size.width, size.height), axisPaint);
    canvas.drawLine(Offset.zero, Offset(0, size.height), axisPaint);
  }

  @override
  bool shouldRepaint(_LossCurvePainter old) => old.history != history;
}

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;

  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 14,
          height: 14,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 4),
        Text(label, style: Theme.of(context).textTheme.labelSmall),
      ],
    );
  }
}
