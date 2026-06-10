// ignore_for_file: lines_longer_than_80_chars

import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

// ─────────────────────────────────────────────────────────────────────────────
// NativeMemory — safe C-memory helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Utility class for safely allocating and copying data between Dart and
/// native (C) memory.
///
/// All allocations are performed through an [Arena] so they are freed
/// automatically when the arena is released:
///
/// ```dart
/// using((arena) {
///   final fPtr = NativeMemory.float32FromList([1.0, 2.0, 3.0], arena);
///   // fPtr is valid inside this block; freed automatically on exit.
/// });
/// ```
class NativeMemory {
  NativeMemory._();

  // ── Float32 ────────────────────────────────────────────────────────────

  /// Copies [data] into a native float32 buffer.
  ///
  /// Returns a `Pointer<Float>` of length `data.length`.
  static Pointer<Float> float32FromList(List<double> data, Arena arena) {
    final ptr = arena.allocate<Float>(data.length * sizeOf<Float>());
    for (int i = 0; i < data.length; i++) {
      ptr[i] = data[i];
    }
    return ptr;
  }

  /// Reads [length] float32 values from [ptr] into a Dart [List<double>].
  static List<double> float32ToList(Pointer<Float> ptr, int length) {
    final out = <double>[];
    for (int i = 0; i < length; i++) {
      out.add(ptr[i]);
    }
    return out;
  }

  /// Copies [data] into a native float32 buffer as a raw [Pointer<Void>].
  static Pointer<Void> float32VoidFromList(List<double> data, Arena arena) =>
      float32FromList(data, arena).cast<Void>();

  // ── Int64 (shapes / labels) ────────────────────────────────────────────

  /// Copies [shape] dimensions into a native int64 array.
  static Pointer<Int64> int64FromList(List<int> values, Arena arena) {
    final ptr = arena.allocate<Int64>(values.length * sizeOf<Int64>());
    for (int i = 0; i < values.length; i++) {
      ptr[i] = values[i];
    }
    return ptr;
  }

  /// Reads [length] int64 values from [ptr].
  static List<int> int64ToList(Pointer<Int64> ptr, int length) {
    final out = <int>[];
    for (int i = 0; i < length; i++) {
      out.add(ptr[i]);
    }
    return out;
  }

  // ── Int32 (class labels in int32 format) ──────────────────────────────

  /// Copies [labels] into a native int32 array (used for class labels).
  static Pointer<Int32> int32FromList(List<int> labels, Arena arena) {
    final ptr = arena.allocate<Int32>(labels.length * sizeOf<Int32>());
    for (int i = 0; i < labels.length; i++) {
      ptr[i] = labels[i];
    }
    return ptr;
  }

  // ── String (C const char*) ─────────────────────────────────────────────

  /// Converts a Dart [String] to a null-terminated UTF-8 `const char*`.
  static Pointer<Void> utf8FromString(String s, Arena arena) =>
      s.toNativeUtf8(allocator: arena).cast<Void>();

  /// Reads a null-terminated C string from [ptr].
  static String stringFromVoid(Pointer<Void> ptr) =>
      ptr.cast<Utf8>().toDartString();

  // ── Float32List (dart:typed_data) ──────────────────────────────────────

  /// Wraps a Dart [Float32List] without copying (zero-copy via asTypedList).
  ///
  /// ⚠ The pointer must remain valid for the lifetime of the returned view.
  static Float32List float32ViewFromPointer(Pointer<Float> ptr, int length) =>
      ptr.asTypedList(length);

  // ── Pointer array (for ORT input/output arrays) ────────────────────────

  /// Allocates a native array of [count] null pointers (Pointer<Pointer<Void>>).
  static Pointer<Pointer<Void>> nullPtrArray(int count, Arena arena) {
    final arr = arena.allocate<Pointer<Void>>(
      count * sizeOf<Pointer<Void>>(),
    );
    for (int i = 0; i < count; i++) {
      arr[i] = nullptr;
    }
    return arr;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// OrtStatusChecker
// ─────────────────────────────────────────────────────────────────────────────

/// Checks an ORT status pointer and throws [OrtNativeException] if non-null.
///
/// Always releases the status object after reading.
///
/// ```dart
/// final status = bindings.createEnv(...);
/// OrtStatusChecker.check(status, bindings);
/// ```
class OrtStatusChecker {
  OrtStatusChecker._();

  static void check(
    Pointer<dynamic> status,
    _OrtStatusApi api,
  ) {
    if (status == nullptr) return; // null == OK in ORT

    final msg = api.getErrorMessage(status);
    final code = api.getErrorCode(status);
    api.releaseStatus(status);

    throw OrtNativeException(
      NativeMemory.stringFromVoid(msg),
      errorCode: code,
    );
  }
}

/// Minimal interface required by [OrtStatusChecker].
abstract class _OrtStatusApi {
  Pointer<Void> getErrorMessage(Pointer<dynamic> status);
  int getErrorCode(Pointer<dynamic> status);
  void releaseStatus(Pointer<dynamic> status);
}

/// Thrown when ONNX Runtime returns a non-OK status from a C call.
class OrtNativeException implements Exception {
  final String message;
  final int errorCode;

  const OrtNativeException(this.message, {this.errorCode = 0});

  @override
  String toString() => 'OrtNativeException [code=$errorCode]: $message';
}
