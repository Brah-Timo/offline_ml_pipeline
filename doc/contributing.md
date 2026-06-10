# Contributing to offline_ml_pipeline

Thank you for your interest in contributing! This document describes the development workflow, code standards, and PR process.

---

## Getting Started

### Prerequisites

- Flutter SDK ≥ 3.10 / Dart SDK ≥ 3.0
- `flutter pub get` in the package root
- For FFI work: ONNX Runtime training library (`libonnxruntime_training.so` or `.dylib`)

### Local Setup

```bash
git clone https://github.com/yourorg/offline_ml_pipeline.git
cd offline_ml_pipeline
flutter pub get
flutter test
flutter analyze
```

---

## Branch Convention

| Branch | Purpose |
|--------|---------|
| `main` | Stable release |
| `genspark_ai_developer` | AI-assisted development |
| `feature/xyz` | Feature branches |
| `fix/xyz` | Bug fix branches |

---

## Coding Standards

### General

- Follow [Effective Dart](https://dart.dev/guides/language/effective-dart) guidelines.
- Use `final` for all immutable variables.
- Prefer named constructors over factory methods for clarity.
- Keep files under 300 lines — split large files by concern.
- Every public API must have a Dart doc comment (`///`).

### FFI Code

- All FFI opaque types must be `final class Foo extends Opaque {}` — never a regular Dart class.
- Never name a Dart class the same as an FFI opaque type in the same scope.
- Always use `Arena`-scoped allocations — never `malloc` directly.
- Release `OrtValue` handles explicitly with `_ort.releaseValue(v)` after use.

### Dart Imports

- All `import` statements must be at the **top of the file**, never inside method bodies.
- Use package imports (`package:offline_ml_pipeline/...`) in tests, never relative `lib/src/...` paths that duplicate the barrel export.

### Error Handling

- Throw typed exceptions from the `PipelineException` hierarchy — never raw `Exception` or `Error`.
- Include contextual information (file path, epoch, step) in exception fields.
- FFI errors: always call `_ort.check(status)` after every ORT C API call.

---

## Adding a New Feature

1. **Create a feature branch**: `git checkout -b feature/my-feature`
2. **Write tests first**: Add unit tests in `test/unit/` before implementation.
3. **Implement**: Keep business logic in `lib/src/`, not in test files.
4. **Run tests**: `flutter test && flutter analyze`
5. **Update docs**: Add/update the relevant `doc/*.md` file.
6. **Commit**: Follow conventional commits format.
7. **PR**: Open a PR to `main` with a description matching the PR template.

---

## Fixing a Bug

1. Reproduce the bug with a failing test.
2. Fix the implementation.
3. Confirm the test now passes.
4. Check that existing tests still pass.
5. Reference the bug in the commit message: `fix(training): resolve NaN loss on zero-variance feature (#42)`.

---

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description (≤72 chars)

Optional longer body explaining why, not what.

Refs: #issue-number
```

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Tests only, no prod code change |
| `refactor` | Code restructure without behaviour change |
| `perf` | Performance improvement |
| `chore` | Build scripts, CI, dependency bumps |

Examples:

```
feat(export): add float16 quantisation for ORT path

fix(ffi): rename OrtTrainingSession Dart class to _OrtBackendSession
to resolve NativeType bound collision with FFI opaque type

docs(contributing): add PR template and coding standards
```

---

## Pull Request Checklist

Before opening a PR:

- [ ] `flutter test` passes (0 failures)
- [ ] `flutter analyze` reports 0 errors, 0 warnings
- [ ] New public APIs have `///` doc comments
- [ ] Relevant `doc/*.md` files updated
- [ ] `CHANGELOG.md` entry added under `## Unreleased`
- [ ] No `import` statements inside method bodies
- [ ] No Dart class names shadowing FFI opaque types

---

## File Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Source files | `snake_case.dart` | `training_session.dart` |
| Test files | `snake_case_test.dart` | `training_metrics_test.dart` |
| Doc files | `snake_case.md` | `ffi_bindings.md` |
| Private classes | `_PascalCase` | `_OrtBackendSession` |
| Public enums | `PascalCase` | `QuantizationMode` |
| Constants | `camelCase` | `kSchemaVersion` |

---

## Running the Full Validation Suite

```bash
# 1. Tests
flutter test

# 2. Static analysis
flutter analyze

# 3. Format check
dart format --output=none --set-exit-if-changed .

# 4. Auto-format (if needed)
dart format .
```

All four must pass before merging.

---

## Versioning

This package follows [Semantic Versioning](https://semver.org/):

- **Patch** (0.0.X): bug fixes, no API changes.
- **Minor** (0.X.0): new features, backward-compatible.
- **Major** (X.0.0): breaking API changes.

Update `version` in `pubspec.yaml` and add a `CHANGELOG.md` entry for every release.

---

## Reporting Issues

Open a GitHub Issue with:

1. Flutter/Dart SDK version (`flutter --version`).
2. Platform (Android, iOS, Linux, Windows, macOS, Web).
3. Minimal reproducible example.
4. Full stack trace (if a crash).
5. Expected vs actual behaviour.
