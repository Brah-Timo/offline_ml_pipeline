import 'dart:isolate';

// ─────────────────────────────────────────────────────────────────────────────
// IsolateRunner
// ─────────────────────────────────────────────────────────────────────────────

/// Runs a computation in a separate Dart [Isolate] to avoid blocking the
/// Flutter UI thread during training.
///
/// Training a neural network (even on small datasets) involves thousands of
/// floating-point operations per second. Running this on the main isolate
/// would freeze animations and gesture responses.
///
/// Usage:
/// ```dart
/// final result = await IsolateRunner.run(() async {
///   // heavy computation here — runs off the main thread
///   return await doSomethingExpensive();
/// });
/// ```
///
/// ## Implementation note
/// Uses the `Isolate.run` API introduced in Dart 2.19 / Flutter 3.7.
/// Falls back gracefully to `compute`-style spawn for older SDK versions.
class IsolateRunner {
  IsolateRunner._();

  /// Executes [computation] in a background isolate and returns the result.
  ///
  /// The [computation] closure must be a **top-level** function or a
  /// static method — Dart's isolate spawning cannot capture instance state.
  ///
  /// For closures that capture local variables, use [runWithMessage].
  static Future<T> run<T>(Future<T> Function() computation) async {
    return Isolate.run(computation);
  }

  /// Spawns an isolate that receives a [message] via a [SendPort].
  ///
  /// Useful when you need to pass a serialisable configuration object to
  /// the background computation.
  ///
  /// ```dart
  /// final result = await IsolateRunner.runWithMessage<Config, String>(
  ///   entryPoint: _backgroundTrain,
  ///   message: config,
  /// );
  /// ```
  static Future<R> runWithMessage<M, R>({
    required _IsolateEntryPoint<M, R> entryPoint,
    required M message,
  }) async {
    final resultPort = ReceivePort();
    final errorPort = ReceivePort();

    await Isolate.spawn(
      _isolateEntry<M, R>,
      _IsolateMessage(
        entryPoint: entryPoint,
        message: message,
        sendPort: resultPort.sendPort,
      ),
      onError: errorPort.sendPort,
    );

    // Wait for either a result or an error
    final result = await Future.any([
      resultPort.first,
      errorPort.first.then((e) {
        final list = e as List;
        throw IsolateException(
          list[0].toString(),
          stackTrace: list[1] != null
              ? StackTrace.fromString(list[1].toString())
              : null,
        );
      }),
    ]);

    resultPort.close();
    errorPort.close();

    return result as R;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

typedef _IsolateEntryPoint<M, R> = Future<R> Function(M message);

class _IsolateMessage<M, R> {
  final _IsolateEntryPoint<M, R> entryPoint;
  final M message;
  final SendPort sendPort;

  const _IsolateMessage({
    required this.entryPoint,
    required this.message,
    required this.sendPort,
  });
}

void _isolateEntry<M, R>(_IsolateMessage<M, R> msg) async {
  final result = await msg.entryPoint(msg.message);
  msg.sendPort.send(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// IsolateException
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps an exception that was thrown inside a background isolate.
class IsolateException implements Exception {
  final String message;
  final StackTrace? stackTrace;

  const IsolateException(this.message, {this.stackTrace});

  @override
  String toString() => 'IsolateException: $message';
}
