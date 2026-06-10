import 'package:flutter/material.dart';
import 'screens/training_screen.dart';

void main() {
  runApp(const OfflineMlExampleApp());
}

class OfflineMlExampleApp extends StatelessWidget {
  const OfflineMlExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'offline_ml_pipeline Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0057FF),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
        ),
      ),
      home: const TrainingScreen(),
    );
  }
}
