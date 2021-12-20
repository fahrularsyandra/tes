import 'package:flutter/services.dart' show rootBundle;
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

Future<void> read() async {
  // final samples = await fromCsv('assets/datasets/games.csv');
  final rawCsvContent =
      await rootBundle.loadString('assets/datasets/diabetes.csv');
  final sample = DataFrame.fromRawCsv(rawCsvContent);
  print(sample);
  final targetColumnName = 'Outcome';
  final splits = splitData(sample, [0.7]);
  final validationData = splits[0];
  print('validationData: $validationData');
  final testData = splits[1];
  print('tesData: $testData');

  final validator = CrossValidator.kFold(sample, numberOfFolds: 5);

  var createClassifier = (DataFrame samples) => LogisticRegressor(
      sample, targetColumnName,
      optimizerType: LinearOptimizerType.gradient,
      iterationsLimit: 90,
      learningRateType: LearningRateType.timeBased,
      batchSize: sample.rows.length,
      probabilityThreshold: 0.7);
  print('cek');
  final scores =
      await validator.evaluate(createClassifier, MetricType.accuracy);
  final accuracy = scores.mean();

  print('accuracy on k fold validation: ${accuracy.toStringAsFixed(2)}');
  final rawCsvTesting =
      await rootBundle.loadString('assets/datasets/tes.csv');
  final tes = DataFrame.fromRawCsv(rawCsvTesting);
  final prediction = createClassifier(sample).predict(tes);
  print('regression: ${prediction.header}');
  for (var item in prediction.rows) {
    print(item);
  }
  var cek = KnnClassifier(sample, 'Outcome', 2);
  var predict = cek.predict(tes);
  print('classification: ${predict.header}');
  for (var item in predict.rows) {
    print(item);
  }
}
