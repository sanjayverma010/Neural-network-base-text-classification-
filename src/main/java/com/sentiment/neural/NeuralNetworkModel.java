package com.sentiment.neural;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Component;

@Component
public class NeuralNetworkModel {

    private MultiLayerNetwork model;
    private int inputSize;
    private int hiddenSize1 = 64;
    private int hiddenSize2 = 32;
    private int outputSize = 1;
    private boolean initialized = false;

    public void initialize(int inputSize) {
        this.inputSize = inputSize;
        buildModel();
        this.initialized = true;
        System.out.println("✅ Neural Network initialized!");
        printModelSummary();
    }

    private void buildModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(hiddenSize1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(hiddenSize1)
                        .nOut(hiddenSize2)
                        .activation(Activation.RELU)
                        .dropOut(0.5)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(hiddenSize2)
                        .nOut(outputSize)
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.XENT) // ✅ FIXED
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
    }

    private void printModelSummary() {
        System.out.println("\n🧠 Neural Network Architecture:");
        System.out.println("┌─────────────────────────────────────────┐");
        System.out.printf("│ Input Layer:     %4d neurons            │\n", inputSize);
        System.out.printf("│ Hidden Layer 1:  %4d neurons (ReLU)     │\n", hiddenSize1);
        System.out.printf("│ Hidden Layer 2:  %4d neurons (ReLU+DO)  │\n", hiddenSize2);
        System.out.printf("│ Output Layer:    %4d neuron (Sigmoid)   │\n", outputSize);
        System.out.println("└─────────────────────────────────────────┘");
    }

    public double predict(double[] features) {
        if (!initialized) {
            System.err.println("⚠️ Model not initialized!");
            return 0.5;
        }

        try {
            INDArray input = Nd4j.create(features, new int[]{1, features.length});
            INDArray output = model.output(input);
            return output.getDouble(0); // sigmoid already applied
        } catch (Exception e) {
            System.err.println("❌ Prediction error: " + e.getMessage());
            return 0.5;
        }
    }

    public void train(double[][] features, double[] labels, int epochs) {
        if (!initialized) {
            System.err.println("⚠️ Model not initialized!");
            return;
        }

        System.out.println("\n🚀 Training Started...");

        INDArray inputArray = Nd4j.create(features);
        INDArray labelArray = Nd4j.create(labels, new int[]{labels.length, 1});

        for (int epoch = 0; epoch < epochs; epoch++) {
            model.fit(inputArray, labelArray);

            if (epoch % 20 == 0 || epoch == epochs - 1) {
                double acc = calculateAccuracy(features, labels);
                System.out.printf("Epoch %d | Accuracy: %.2f%%%n", epoch, acc * 100);
            }
        }

        System.out.println("✅ Training Completed!");
    }

    private double calculateAccuracy(double[][] features, double[] labels) {
        int correct = 0;

        for (int i = 0; i < features.length; i++) {
            double pred = predict(features[i]);
            int p = pred > 0.5 ? 1 : 0;
            int a = labels[i] > 0.5 ? 1 : 0;

            if (p == a) correct++;
        }

        return correct / (double) features.length;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }
}