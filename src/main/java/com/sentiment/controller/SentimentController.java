package com.sentiment.controller;

import com.sentiment.model.SentimentRequest;
import com.sentiment.model.SentimentResponse;
import com.sentiment.service.SentimentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/sentiment")
@CrossOrigin(origins = "*")
public class SentimentController {
    
    @Autowired
    private SentimentService sentimentService;
    
    @PostMapping("/analyze")
    public ResponseEntity<SentimentResponse> analyzeSentiment(@RequestBody SentimentRequest request) {
        String inputText = request != null ? request.getText() : null;
        System.out.println("📝 Analyzing: " + inputText);
        SentimentResponse response = sentimentService.analyzeSentiment(inputText);
        System.out.println("🎯 Result: " + response.getSentiment() + " (Confidence: " + 
                          String.format("%.2f", response.getConfidence() * 100) + "%)");
        return ResponseEntity.ok(response);
    }
    
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> status = new HashMap<>();
        status.put("status", "OK");
        status.put("message", "Sentiment Analysis Service is running");
        status.put("timestamp", System.currentTimeMillis());
        status.put("model", "Neural Network (DeepLearning4J)");
        return ResponseEntity.ok(status);
    }
    
    @PostMapping("/batch")
    public ResponseEntity<Map<String, SentimentResponse>> analyzeBatch(@RequestBody Map<String, String> requests) {
        Map<String, SentimentResponse> responses = new HashMap<>();
        
        for (Map.Entry<String, String> entry : requests.entrySet()) {
            SentimentResponse response = sentimentService.analyzeSentiment(entry.getValue());
            responses.put(entry.getKey(), response);
        }
        
        return ResponseEntity.ok(responses);
    }
    
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("vocabularySize", sentimentService.getVocabularySize());
        stats.put("modelInitialized", sentimentService.isModelInitialized());
        stats.put("totalPredictions", sentimentService.getTotalPredictions());
        return ResponseEntity.ok(stats);
    }
}