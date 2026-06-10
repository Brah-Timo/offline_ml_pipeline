// ignore_for_file: use_build_context_synchronously, unused_field

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:offline_ml_pipeline/offline_ml_pipeline.dart';
import '../widgets/metrics_card.dart';
import '../widgets/epoch_chart.dart';

/// The main screen — lets the user pick a CSV, configure training,
/// watch a live progress bar, and see the final metrics.
class TrainingScreen extends StatefulWidget {
  const TrainingScreen({super.key});

  @override
  State<TrainingScreen> createState() => _TrainingScreenState();
}

class _TrainingScreenState extends State<TrainingScreen> {
  // ── State ────────────────────────────────────────────────────────────────
  String? _csvPath;
  String _targetColumn = 'label';
  ModelType _modelType = ModelType.classifier;
  int _epochs = 50;

  bool _isTraining = false;
  double _progress = 0.0;
  String _statusText = 'Pick a CSV file to get started.';
  TrainingProgress? _latestProgress;
  PipelineResult? _result;
  String? _errorText;

  // ── CSV picker ────────────────────────────────────────────────────────────

  Future<void> _pickCsv() async {
    final pick = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['csv'],
    );
    if (pick != null && pick.files.single.path != null) {
      setState(() {
        _csvPath = pick.files.single.path;
        _statusText = 'Selected: ${pick.files.single.name}';
        _result = null;
        _errorText = null;
      });
    }
  }

  // ── Training ──────────────────────────────────────────────────────────────

  Future<void> _startTraining() async {
    if (_csvPath == null) return;

    setState(() {
      _isTraining = true;
      _progress = 0.0;
      _result = null;
      _errorText = null;
      _statusText = 'Initialising…';
    });

    final pipeline = MlPipeline(
      config: PipelineConfig(
        csvPath: _csvPath!,
        targetColumn: _targetColumn.trim(),
        modelType: _modelType,
        epochs: _epochs,
        batchSize: 32,
        lossFunction: _modelType == ModelType.classifier
            ? LossFunction.crossEntropy
            : LossFunction.mse,
        earlyStopping: const EarlyStopping(patience: 8),
      ),
    );

    // Live progress updates
    pipeline.progressStream.listen((p) {
      setState(() {
        _latestProgress = p;
        _progress = p.percentage / 100.0;
        _statusText = 'Epoch ${p.epoch}/${p.totalEpochs}  '
            '| loss ${p.trainLoss.toStringAsFixed(4)}';
      });
    });

    try {
      final result = await pipeline.train();
      setState(() {
        _result = result;
        _isTraining = false;
        _progress = 1.0;
        _statusText = 'Training complete!';
      });
    } on CsvFileNotFoundException catch (e) {
      _handleError('CSV not found: ${e.csvPath}');
    } on ColumnNotFoundException catch (e) {
      _handleError('Column "${e.column}" not found. '
          'Available: ${e.availableColumns.join(", ")}');
    } on OrtException catch (e) {
      _handleError('ORT error: ${e.message}');
    } catch (e) {
      _handleError(e.toString());
    }
  }

  void _handleError(String msg) {
    setState(() {
      _isTraining = false;
      _errorText = msg;
      _statusText = 'Error — see details below.';
    });
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('offline_ml_pipeline'),
        centerTitle: true,
        backgroundColor: theme.colorScheme.primaryContainer,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── CSV file picker ─────────────────────────────────────────
            _SectionHeader('1. Select CSV File'),
            const SizedBox(height: 8),
            OutlinedButton.icon(
              onPressed: _isTraining ? null : _pickCsv,
              icon: const Icon(Icons.upload_file_rounded),
              label: Text(
                _csvPath != null
                    ? _csvPath!.split('/').last
                    : 'Choose CSV…',
              ),
            ),

            const SizedBox(height: 24),

            // ── Configuration ───────────────────────────────────────────
            _SectionHeader('2. Configure Training'),
            const SizedBox(height: 8),

            // Target column
            TextField(
              decoration: const InputDecoration(
                labelText: 'Target column name',
                hintText: 'e.g. species, price, diagnosis',
              ),
              onChanged: (v) => setState(() => _targetColumn = v),
              controller: TextEditingController(text: _targetColumn),
            ),
            const SizedBox(height: 12),

            // Model type
            Row(
              children: [
                const Text('Model type:'),
                const SizedBox(width: 16),
                ChoiceChip(
                  label: const Text('Classifier'),
                  selected: _modelType == ModelType.classifier,
                  onSelected: _isTraining
                      ? null
                      : (_) => setState(
                            () => _modelType = ModelType.classifier,
                          ),
                ),
                const SizedBox(width: 8),
                ChoiceChip(
                  label: const Text('Regressor'),
                  selected: _modelType == ModelType.regressor,
                  onSelected: _isTraining
                      ? null
                      : (_) => setState(
                            () => _modelType = ModelType.regressor,
                          ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Epochs slider
            Row(
              children: [
                Text('Epochs: $_epochs'),
                Expanded(
                  child: Slider(
                    value: _epochs.toDouble(),
                    min: 10,
                    max: 200,
                    divisions: 19,
                    label: '$_epochs',
                    onChanged: _isTraining
                        ? null
                        : (v) => setState(() => _epochs = v.round()),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 24),

            // ── Train button ────────────────────────────────────────────
            _SectionHeader('3. Train'),
            const SizedBox(height: 8),
            FilledButton.icon(
              onPressed: (_csvPath != null && !_isTraining)
                  ? _startTraining
                  : null,
              icon: _isTraining
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.play_arrow_rounded),
              label: Text(_isTraining ? 'Training…' : 'Train On-Device'),
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(52),
              ),
            ),

            const SizedBox(height: 16),

            // ── Progress ────────────────────────────────────────────────
            LinearProgressIndicator(
              value: _progress,
              minHeight: 8,
              borderRadius: BorderRadius.circular(4),
            ),
            const SizedBox(height: 8),
            Text(
              _statusText,
              style: theme.textTheme.bodySmall
                  ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
            ),

            // ── Error ───────────────────────────────────────────────────
            if (_errorText != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _errorText!,
                  style: TextStyle(color: theme.colorScheme.error),
                ),
              ),
            ],

            // ── Results ─────────────────────────────────────────────────
            if (_result != null) ...[
              const SizedBox(height: 24),
              _SectionHeader('4. Results'),
              const SizedBox(height: 8),
              MetricsCard(result: _result!, modelType: _modelType),
              const SizedBox(height: 16),
              EpochChart(history: _result!.epochHistory),
              const SizedBox(height: 12),
              _ModelFileTile(result: _result!),
            ],

            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-widgets
// ─────────────────────────────────────────────────────────────────────────────

class _SectionHeader extends StatelessWidget {
  final String text;
  const _SectionHeader(this.text);

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
    );
  }
}

class _ModelFileTile extends StatelessWidget {
  final PipelineResult result;
  const _ModelFileTile({required this.result});

  @override
  Widget build(BuildContext context) {
    return ListTile(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      leading: const Icon(Icons.model_training_rounded),
      title: Text(result.modelPath.split('/').last),
      subtitle: Text(
        'Backend: ${result.backend}  •  '
        '${result.epochHistory.length} epochs  •  '
        '${result.trainingDuration.inSeconds}s',
      ),
      trailing: const Icon(Icons.check_circle_rounded, color: Colors.green),
    );
  }
}
