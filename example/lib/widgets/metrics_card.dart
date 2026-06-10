import 'package:flutter/material.dart';
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';

/// Displays final training metrics in a card.
class MetricsCard extends StatelessWidget {
  final PipelineResult result;
  final ModelType modelType;

  const MetricsCard({
    super.key,
    required this.result,
    required this.modelType,
  });

  @override
  Widget build(BuildContext context) {
    final m = result.metrics;

    final rows = <_MetricRow>[];

    rows.add(_MetricRow(
      'Train Loss',
      m.finalTrainLoss.toStringAsFixed(4),
    ));
    rows.add(_MetricRow(
      'Val Loss',
      m.finalValLoss >= 0
          ? m.finalValLoss.toStringAsFixed(4)
          : '—',
    ));

    if (modelType == ModelType.classifier) {
      if (m.accuracy != null) {
        rows.add(_MetricRow(
          'Accuracy',
          '${(m.accuracy! * 100).toStringAsFixed(2)} %',
          highlight: true,
        ));
      }
      if (m.f1Score != null) {
        rows.add(_MetricRow('F1 Score', m.f1Score!.toStringAsFixed(4)));
      }
      if (m.precision != null) {
        rows.add(_MetricRow('Precision', m.precision!.toStringAsFixed(4)));
      }
      if (m.recall != null) {
        rows.add(_MetricRow('Recall', m.recall!.toStringAsFixed(4)));
      }
    } else {
      if (m.rmse != null) {
        rows.add(
          _MetricRow('RMSE', m.rmse!.toStringAsFixed(4), highlight: true),
        );
      }
      if (m.mae != null) {
        rows.add(_MetricRow('MAE', m.mae!.toStringAsFixed(4)));
      }
      if (m.rSquared != null) {
        rows.add(_MetricRow('R²', m.rSquared!.toStringAsFixed(4)));
      }
      if (m.mape != null) {
        rows.add(
          _MetricRow(
              'MAPE', '${(m.mape! * 100).toStringAsFixed(2)} %'),
        );
      }
    }

    if (m.stoppedEarly) {
      rows.add(const _MetricRow('Early stopped', 'Yes'));
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Column(
          children: [
            for (final row in rows)
              ListTile(
                dense: true,
                title: Text(row.label),
                trailing: Text(
                  row.value,
                  style: TextStyle(
                    fontWeight: row.highlight
                        ? FontWeight.bold
                        : FontWeight.normal,
                    color: row.highlight
                        ? Theme.of(context).colorScheme.primary
                        : null,
                    fontSize: row.highlight ? 16 : 14,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _MetricRow {
  final String label;
  final String value;
  final bool highlight;

  const _MetricRow(this.label, this.value, {this.highlight = false});
}
