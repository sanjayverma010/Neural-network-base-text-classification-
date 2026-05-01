package com.sentiment.model;

public class SentimentResponse {
    private String text;
    private String sentiment;
    private double confidence;
    private double score;
    private long timestamp;
    
    public SentimentResponse() {}
    
    public SentimentResponse(String text, String sentiment, double confidence, double score) {
        this.text = text;
        this.sentiment = sentiment;
        this.confidence = confidence;
        this.score = score;
        this.timestamp = System.currentTimeMillis();
    }
    
    // Getters and Setters
    public String getText() { return text; }
    public void setText(String text) { this.text = text; }
    
    public String getSentiment() { return sentiment; }
    public void setSentiment(String sentiment) { this.sentiment = sentiment; }
    
    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
    
    public double getScore() { return score; }
    public void setScore(double score) { this.score = score; }
    
    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
}