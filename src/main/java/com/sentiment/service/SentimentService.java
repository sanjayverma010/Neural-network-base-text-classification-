package com.sentiment.service;

import com.sentiment.model.SentimentResponse;
import com.sentiment.neural.NeuralNetworkModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.concurrent.atomic.AtomicLong;

@Service
public class SentimentService {
    
    @Autowired
    private NeuralNetworkModel neuralNetwork;
    
    @Autowired
    private DataPreprocessingService preprocessingService;
    
    private AtomicLong totalPredictions = new AtomicLong(0);
    
    public SentimentResponse analyzeSentiment(String text) {
        totalPredictions.incrementAndGet();
        
        if (text == null || text.trim().isEmpty()) {
            return new SentimentResponse(text, "Neutral", 0.5, 0.5);
        }
        
        try {
            // Step 1: Preprocess text
            String cleanedText = preprocessingService.preprocessText(text);
            
            // Step 2: Convert to numerical vector (Bag of Words)
            double[] features = preprocessingService.textToVector(cleanedText);
            
            // Step 3: Neural Network prediction
            double prediction = neuralNetwork.predict(features);
            
            // Step 4: Interpret the result as text classes.
            // We keep a neutral band in the center to avoid forcing weak predictions.
            String sentiment;
            double confidence;
            if (prediction >= 0.6) {
                sentiment = "Positive";
                confidence = prediction;
            } else if (prediction <= 0.4) {
                sentiment = "Negative";
                confidence = 1 - prediction;
            } else {
                sentiment = "Neutral";
                // Neutral confidence increases when score is close to 0.5
                confidence = 1 - (Math.abs(prediction - 0.5) * 5);
            }
            
            // Round to 4 decimal places
            confidence = Math.round(confidence * 10000.0) / 10000.0;
            prediction = Math.round(prediction * 10000.0) / 10000.0;
            
            System.out.printf("📊 Text: '%s' → Score: %.4f → %s (%.1f%% confidence)%n", 
                              text, prediction, sentiment, confidence * 100);
            
            return new SentimentResponse(text, sentiment, confidence, prediction);
            
        } catch (Exception e) {
            System.err.println("❌ Error analyzing sentiment: " + e.getMessage());
            e.printStackTrace();
            return new SentimentResponse(text, "Error", 0.0, 0.0);
        }
    }
    
    public double getSentimentScore(String text) {
        String cleanedText = preprocessingService.preprocessText(text);
        double[] features = preprocessingService.textToVector(cleanedText);
        return neuralNetwork.predict(features);
    }
    
    public int getVocabularySize() {
        return preprocessingService.getVocabularySize();
    }
    
    public boolean isModelInitialized() {
        return neuralNetwork.isInitialized();
    }
    
    public long getTotalPredictions() {
        return totalPredictions.get();
    }
}