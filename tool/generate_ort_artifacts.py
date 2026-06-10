#!/usr/bin/env python3
"""
generate_ort_artifacts.py
──────────────────────────
Generates ONNX Runtime On-Device Training artifacts for offline_ml_pipeline.

Run this script **once** on a development machine (not on-device) to produce
the four training artifact files:
    training_model.onnx
    eval_model.onnx
    optimizer_model.onnx
    checkpoint

These are then bundled into assets/ort_training_templates/ and shipped with
the app. On the device, the Dart layer loads them via TrainingSession.create().

Requirements:
    pip install onnxruntime-training torch onnx numpy

Usage:
    # Classifier with 4 features, 3 classes (e.g., Iris dataset)
    python tool/generate_ort_artifacts.py \\
        --model_type classifier \\
        --features 4 \\
        --classes 3 \\
        --output assets/ort_training_templates/classifier_4f_3c

    # Regressor with 10 features
    python tool/generate_ort_artifacts.py \\
        --model_type regressor \\
        --features 10 \\
        --output assets/ort_training_templates/regressor_10f
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import onnxruntime.training.artifacts as artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp_classifier(input_features: int, num_classes: int) -> nn.Module:
    """Shallow MLP for classification (mirrors ModelArchitecture.mlpShallow)."""
    return nn.Sequential(
        nn.Linear(input_features, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes),
        # Note: CrossEntropyLoss includes LogSoftmax, so no Softmax here.
    )


def build_mlp_deep_classifier(input_features: int, num_classes: int) -> nn.Module:
    """Deeper MLP (ModelArchitecture.mlpDeep)."""
    return nn.Sequential(
        nn.Linear(input_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, num_classes),
    )


def build_mlp_regressor(input_features: int) -> nn.Module:
    """Shallow MLP for regression (single scalar output)."""
    return nn.Sequential(
        nn.Linear(input_features, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def build_linear_classifier(input_features: int, num_classes: int) -> nn.Module:
    """Single linear layer — fastest to train."""
    return nn.Linear(input_features, num_classes)


def build_linear_regressor(input_features: int) -> nn.Module:
    """Single linear layer for regression."""
    return nn.Linear(input_features, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_artifacts(
    model: nn.Module,
    input_features: int,
    output_dir: str,
    loss_type: str = 'CrossEntropyLoss',
    optimizer_type: str = 'AdamW',
    opset_version: int = 17,
) -> None:
    """Export the model to ONNX then generate ORT training artifacts."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Export forward model to ONNX ─────────────────────────────
    model.eval()
    dummy_input = torch.randn(1, input_features)
    onnx_path = os.path.join(output_dir, 'forward_model.onnx')

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        opset_version=opset_version,
        export_params=True,
        training=torch.onnx.TrainingMode.TRAINING,
        do_constant_folding=False,
    )
    print(f'  [✓] Forward ONNX model → {onnx_path}')

    # ── Step 2: Generate ORT training artifacts ───────────────────────────
    requires_grad = [name for name, _ in model.named_parameters()]

    opt_type = getattr(artifacts.OptimType, optimizer_type)
    loss_type_val = getattr(artifacts.LossType, loss_type)

    artifacts.generate_artifacts(
        onnx_path,
        optimizer=opt_type,
        loss=loss_type_val,
        artifact_directory=output_dir,
        requires_grad=requires_grad,
    )

    print(f'  [✓] ORT training artifacts → {output_dir}')
    print(f'      training_model.onnx  — used for TrainStep')
    print(f'      eval_model.onnx      — used for EvalStep')
    print(f'      optimizer_model.onnx — used for OptimizerStep')
    print(f'      checkpoint           — trainable parameters state')

    # ── Step 3: Clean up intermediate file ───────────────────────────────
    os.remove(onnx_path)
    print(f'  [✓] Cleaned up forward_model.onnx')


# ─────────────────────────────────────────────────────────────────────────────
# Bulk template generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_templates(base_output_dir: str) -> None:
    """
    Generates a set of pre-built templates for common configurations.
    These cover the most frequent use-cases out of the box:

        Classifiers: 4f×2c, 4f×3c, 8f×2c, 16f×5c, 32f×10c
        Regressors:  4f, 8f, 16f, 32f
    """
    configs = [
        # (model_type, features, classes)
        ('classifier', 4, 2),
        ('classifier', 4, 3),
        ('classifier', 8, 2),
        ('classifier', 8, 4),
        ('classifier', 16, 5),
        ('classifier', 32, 10),
        ('regressor', 4, 1),
        ('regressor', 8, 1),
        ('regressor', 16, 1),
        ('regressor', 32, 1),
    ]

    for model_type, features, classes in configs:
        tag = (f'classifier_{features}f_{classes}c'
               if model_type == 'classifier'
               else f'regressor_{features}f')
        out_dir = os.path.join(base_output_dir, tag)
        print(f'\nGenerating {tag}…')

        if model_type == 'classifier':
            model = build_mlp_classifier(features, classes)
            generate_artifacts(
                model, features, out_dir,
                loss_type='CrossEntropyLoss',
                optimizer_type='AdamW',
            )
        else:
            model = build_mlp_regressor(features)
            generate_artifacts(
                model, features, out_dir,
                loss_type='MSELoss',
                optimizer_type='AdamW',
            )

    print('\n[✓] All templates generated successfully.')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate ORT On-Device Training artifacts.',
    )
    parser.add_argument(
        '--model_type',
        choices=['classifier', 'regressor', 'all'],
        default='all',
        help='Model type to generate. "all" generates a standard set of templates.',
    )
    parser.add_argument('--features', type=int, default=4,
                        help='Number of input features.')
    parser.add_argument('--classes', type=int, default=3,
                        help='Number of output classes (classifier only).')
    parser.add_argument(
        '--architecture',
        choices=['linear', 'shallow', 'deep'],
        default='shallow',
        help='Network depth.',
    )
    parser.add_argument(
        '--output',
        default='assets/ort_training_templates',
        help='Output directory for artifacts.',
    )
    args = parser.parse_args()

    # Verify onnxruntime-training is installed
    try:
        import onnxruntime.training.artifacts as _
    except ImportError:
        print('ERROR: onnxruntime-training not installed.')
        print('Run: pip install onnxruntime-training torch onnx')
        sys.exit(1)

    if args.model_type == 'all':
        generate_all_templates(args.output)
        return

    print(f'\nGenerating {args.model_type} artifacts…')
    print(f'  features  = {args.features}')
    if args.model_type == 'classifier':
        print(f'  classes   = {args.classes}')
    print(f'  output    = {args.output}')
    print()

    if args.model_type == 'classifier':
        if args.architecture == 'linear':
            model = build_linear_classifier(args.features, args.classes)
        elif args.architecture == 'deep':
            model = build_mlp_deep_classifier(args.features, args.classes)
        else:
            model = build_mlp_classifier(args.features, args.classes)

        generate_artifacts(
            model, args.features, args.output,
            loss_type='CrossEntropyLoss',
        )
    else:
        if args.architecture == 'linear':
            model = build_linear_regressor(args.features)
        else:
            model = build_mlp_regressor(args.features)

        generate_artifacts(
            model, args.features, args.output,
            loss_type='MSELoss',
        )

    print('\n[✓] Done.')


if __name__ == '__main__':
    main()
