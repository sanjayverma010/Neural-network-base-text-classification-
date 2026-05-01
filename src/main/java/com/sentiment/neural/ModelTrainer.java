package com.sentiment.neural;

import java.util.ArrayList;
import java.util.List;

import com.sentiment.service.DataPreprocessingService;
import com.sentiment.neural.NeuralNetworkModel;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;

@Component
public class ModelTrainer {

    @Autowired
    private NeuralNetworkModel neuralNetwork;

    @Autowired
    private DataPreprocessingService preprocessingService;

    @PostConstruct
    public void trainModel() {
        System.out.println("\n📚 Preparing training dataset...");

        // Prepare training data
        List<TrainingExample> trainingData = new ArrayList<>();

        // POSITIVE examples (label = 1.0)
        String[] positiveExamples = {
                "This movie is amazing", "great product very good", "I love this",
                "excellent work wonderful", "best experience ever", "fantastic quality",
                "awesome performance", "really good value", "perfect solution",
                "beautiful design", "very happy with this", "outstanding result",
                "recommend to everyone", "exceeded my expectations", "brilliant work"
        };

        // NEGATIVE examples (label = 0.0)
        String[] negativeExamples = {
                "worst product ever", "terrible experience bad", "I hate this",
                "very disappointing poor quality", "waste of money",
                "awful customer service", "horrible design", "really bad product",
                "useless feature", "mediocre performance", "annoying interface",
                "terrible value for money", "never buy again", "complete disaster"
        };

        for (String text : positiveExamples) {
            trainingData.add(new TrainingExample(text, 1.0));
        }

        for (String text : negativeExamples) {
            trainingData.add(new TrainingExample(text, 0.0));
        }

        // Mixed examples
        trainingData.add(new TrainingExample("not bad could be better", 0.4));
        trainingData.add(new TrainingExample("quite good actually", 0.7));
        trainingData.add(new TrainingExample("average nothing special", 0.5));
        trainingData.add(new TrainingExample("better than expected", 0.8));
        trainingData.add(new TrainingExample("could be worse", 0.3));
        trainingData.add(new TrainingExample("satisfied with purchase", 0.9));
        trainingData.add(new TrainingExample("not worth the price", 0.2));

        System.out.println("📊 Total training samples: " + trainingData.size());

        // Convert to arrays for training
        int vocabSize = preprocessingService.getVocabularySize();
        double[][] features = new double[trainingData.size()][vocabSize];
        double[] labels = new double[trainingData.size()];

        for (int i = 0; i < trainingData.size(); i++) {
            TrainingExample example = trainingData.get(i);
            String processedText = preprocessingService.preprocessText(example.text);
            features[i] = preprocessingService.textToVector(processedText);
            labels[i] = example.label;
        }

        // Initialize and train the neural network
        neuralNetwork.initialize(vocabSize);
        neuralNetwork.train(features, labels, 100);

        // Test the model
        testModel();
    }

    private void testModel() {
        System.out.println("\n🧪 Testing Model on New Examples:");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        String[][] testCases = {
                { "I absolutely love this product!", "Should be Positive" },
                { "This is terrible, worst purchase ever", "Should be Negative" },
                { "Pretty good, I'm satisfied", "Should be Positive" },
                { "Not great, but okay", "Should be Neutral/Negative" },
                { "Amazing quality, highly recommended", "Should be Positive" },
                { "Waste of money, very disappointed", "Should be Negative" },
                { "Could be better but it's fine", "Should be Neutral" },
                { "Best thing I ever bought!", "Should be Positive" }
        };

        for (String[] testCase : testCases) {
            String text = testCase[0];
            String expected = testCase[1];

            double[] vector = preprocessingService.textToVector(text);
            double prediction = neuralNetwork.predict(vector);
            String sentiment = prediction > 0.5 ? "Positive" : "Negative";
            double confidence = prediction > 0.5 ? prediction : 1 - prediction;

            String emoji = sentiment.equals("Positive") ? "😊" : "😞";
            System.out.printf("%s Text: \"%s\"%n", emoji, text);
            System.out.printf("   Prediction: %.4f → %s (%.1f%% confidence)%n",
                    prediction, sentiment, confidence * 100);
            System.out.printf("   Expected: %s%n", expected);
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }
    }

    private static class TrainingExample {
        String text;
        double label;

        TrainingExample(String text, double label) {
            this.text = text;
            this.label = label;
        }
    }
}