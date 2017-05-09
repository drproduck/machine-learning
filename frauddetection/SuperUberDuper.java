import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.Predictor;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by drproduck on 4/30/17.
 */
public class SuperUberDuper {

    public static long start;
    public static long stop;
    public static void main(String[] args) throws IOException {
        writer = new BufferedWriter(new FileWriter("/home/drproduck/Documents/result.txt"));

        SparkSession spark = SparkSession
                .builder()
                .appName("superduper").config("spark.master", "local") //remove this config to run in cluster
                .getOrCreate();

        //load dataset
        Dataset<Row> dataFrame = spark.read().option("header", "true").option("inferSchema", "true").csv("/home/drproduck/Documents/creditcard.csv");
        dataFrame.show(5);

        List<Triple<String, Long, Double>> list = new ArrayList<>();

        //group feature column into 1
        String[] columns = dataFrame.columns();
        String[] features = Arrays.copyOfRange(columns, 0, columns.length - 1);
        VectorAssembler assembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
        dataFrame = assembler.transform(dataFrame);

        //split dataset into train and test with ratio 50:50. Read TODO
        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.5, 0.5});
        trainingData = splits[0];
        testData = splits[1];

        //random forest
        RandomForestClassifier random_forest = new RandomForestClassifier().setLabelCol("Class").setFeaturesCol("features");
        start = System.nanoTime();
        RandomForestClassificationModel rfModel = random_forest.fit(trainingData);
        Dataset<Row> predictions = rfModel.transform(testData);
        stop = System.nanoTime();
        predictions.select("Class", "prediction").show(5);//predictions at last few columns

        //cast int to double to use with confusion matrix
        predictions = predictions.withColumn("Class", predictions.col("Class").cast(DataTypes.DoubleType));

        writer.write("Random forest and gradient boosting with all features \n");
        //confusion matrix (precision - recall, can use F1)
        MulticlassMetrics metrics = new MulticlassMetrics(predictions.select("prediction", "Class"));
        writer.write("random forest with 30 features: \n\n"+metrics.confusionMatrix().toString()+"\n");
        writer.write("run time: " + (stop - start) + "\n");
        writer.write("F1 score: " + metrics.fMeasure() + "\n");
        writer.newLine();
        list.add(new Triple<String, Long, Double>("random forest with 30 features", stop - start, metrics.fMeasure()));

        GBTClassifier gradient_boosting = new GBTClassifier().setLabelCol("Class").setFeaturesCol("features");

        list.add(train_and_test(gradient_boosting, "gradient tree boosting with 30 features"));
        list.add(train_and_test(new DecisionTreeClassifier().setLabelCol("Class").setFeaturesCol("features"), "decision tree with 30 features"));
        list.add(train_and_test(new MultilayerPerceptronClassifier().setLabelCol("Class").setFeaturesCol("features").setLayers(new int[]{30, 30, 2}).setBlockSize(150).setMaxIter(50), "neural network (10, 30, 30, 2) with 30 features"));

        //ranking of feature importance
        ArrayList<Pair<Integer, Double>> rank = new ArrayList<>();
        int i = 0;
        for (Double d : rfModel.featureImportances().toArray()) {
            rank.add(new Pair(i++, d));
        }
        Collections.sort(rank, (Pair a, Pair b) -> -1 * Double.compare((double) a.second, (double) b.second));
        Iterator<Pair<Integer, Double>> iter = rank.listIterator();
        String[] a = new String[rank.size()];
        i = 0;
        writer.write("Ranking of feature importance\n");
        while (iter.hasNext()) {
            Pair<Integer, Double> p = iter.next();
            a[i] = (dataFrame.columns()[p.first]);
            writer.write(a[i] + ", " + p.second+"\n");
            i++;
        }

        //truncate dataset to use only top 10 features
        features = new String[]{"V7", "V10", "V18", "V4", "V9", "V16", "V14", "V11", "V17", "V12"};
        Dataset<Row> truncatedDataFrame = dataFrame.select("V7", "V10", "V18", "V4", "V9", "V16", "V14", "V11", "V17", "V12", "Class");

        truncatedDataFrame.show(5);

        assembler = new VectorAssembler().setInputCols(features).setOutputCol("top_features");

        DecisionTreeClassifier decision_tree = new DecisionTreeClassifier().setLabelCol("Class").setFeaturesCol("top_features");
        random_forest = new RandomForestClassifier().setLabelCol("Class").setFeaturesCol("top_features");
        NaiveBayes naive_bayes = new NaiveBayes().setLabelCol("Class").setFeaturesCol("top_features");
        gradient_boosting = new GBTClassifier().setLabelCol("Class").setFeaturesCol("top_features");
        MultilayerPerceptronClassifier neural_net = new MultilayerPerceptronClassifier().setLabelCol("Class").setFeaturesCol("top_features").setLayers(new int[]{10, 30, 2}).setBlockSize(150).setMaxIter(50);

        truncatedDataFrame = assembler.transform(truncatedDataFrame);

        splits = truncatedDataFrame.randomSplit(new double[]{0.5, 0.5});
        trainingData = splits[0];
        testData = splits[1];

        writer.newLine();



        list.add(train_and_test(decision_tree, "decision tree with 10 features"));
        list.add(train_and_test(random_forest, "random forest with 10 features"));
//        train_and_test(nb, "Naive Bayes");
        list.add(train_and_test(gradient_boosting, "gradient tree boosting with 10 features"));
        list.add(train_and_test(neural_net, "neural network (10, 30, 2) with 10 features"));
        writer.write("\nNo significant improvement?\n");
        Collections.sort(list, (Triple<String, Long, Double> c, Triple<String, Long, Double> b) -> Long.compare(c.b, b.b));
        writer.newLine();
        writer.write("methods sorted by run time: \n");
        for (Triple x : list) {
            writer.write(x.a + "\t" + x.b + "\t" + x.c + "\n");
        }
        writer.write("--------------------------------------");
        writer.newLine();
        Collections.sort(list, (Triple<String, Long, Double> c, Triple<String, Long, Double> b) -> Double.compare(c.c, b.c));
        writer.write("methods sorted by F1 score: \n");
        for (Triple x : list) {
            writer.write(x.a + "\t" + x.b + "\t" + x.c + "\n");
        }
        writer.close();

    }

    public static Dataset<Row> testData;
    public static Dataset<Row> trainingData;
    public static Dataset<Row> predictions;
    public static BufferedWriter writer;

    public static Triple<String, Long, Double> train_and_test(Predictor classifier, String name) throws IOException {

        start = System.nanoTime();
        PredictionModel classificationModel = (PredictionModel) classifier.fit(trainingData);
        predictions = classificationModel.transform(testData);
        stop = System.nanoTime();
        predictions.show(5);
        predictions.printSchema();

        predictions = predictions.withColumn("Class", predictions.col("Class").cast(DataTypes.DoubleType));
        predictions.show(5);

        MulticlassMetrics metrics = new MulticlassMetrics(predictions.select("prediction", "Class"));

        writer.write(name+"\n\n");
        writer.write("run time: " + (stop - start));
        writer.newLine();
        writer.write("Confusion matrix: \n"+metrics.confusionMatrix().toString()+"\n");
        double[] matrix = metrics.confusionMatrix().toArray();
        System.out.println(Arrays.toString(matrix));

        writer.write("true positive ratio: " + matrix[0] / (matrix[0] + matrix[2]) + "\n");
        writer.write("true negative ratio: " + matrix[3] / (matrix[1] + matrix[3]) + "\n");
        writer.write("F1 score: "+metrics.fMeasure());
        writer.newLine();
        writer.write("--------------------------------------");
        writer.newLine();
        return new Triple<String, Long, Double>(name, stop - start, metrics.fMeasure());
    }
}

class Pair<A, B> {
    A first;
    B second;

    Pair(A f, B s) {
        first = f;
        second = s;
    }
}

class Triple<A, B, C> {
    A a;
    B b;
    C c;

    Triple(A a, B b, C c) {
        this.a = a;
        this.b = b;
        this.c = c;
    }
}
