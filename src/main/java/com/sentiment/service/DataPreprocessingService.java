package com.sentiment.service;

import org.springframework.stereotype.Service;
import java.util.*;
import java.util.regex.Pattern;

@Service
public class DataPreprocessingService {
    
    private static final Pattern SPECIAL_CHARS = Pattern.compile("[^a-zA-Z\\s]");
    private static final Pattern MULTIPLE_SPACES = Pattern.compile("\\s+");
    
    private Map<String, Integer> vocabulary;
    private List<String> wordList;
    
    public DataPreprocessingService() {
        this.vocabulary = new HashMap<>();
        this.wordList = new ArrayList<>();
        initializeVocabulary();
    }
    
    private void initializeVocabulary() {
        // Positive words
        String[] positiveWords = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
            "awesome", "love", "like", "enjoy", "best", "nice", "perfect",
            "beautiful", "brilliant", "superb", "outstanding", "positive",
            "happy", "satisfied", "recommend", "brilliant", "incredible"
        };
        
        // Negative words
        String[] negativeWords = {
            "bad", "terrible", "awful", "horrible", "worst", "poor", "hate",
            "dislike", "boring", "useless", "waste", "disappointing", 
            "mediocre", "negative", "annoying", "frustrating", "stupid",
            "ridiculous", "pathetic", "useless", "terrible"
        };
        
        for (String word : positiveWords) {
            addWordToVocabulary(word);
        }
        
        for (String word : negativeWords) {
            addWordToVocabulary(word);
        }
        
        // Common words
        String[] commonWords = {"the", "a", "an", "and", "or", "but", "so", 
                                "very", "really", "quite", "some", "more"};
        
        for (String word : commonWords) {
            addWordToVocabulary(word);
        }
        
        System.out.println("📚 Vocabulary initialized with " + vocabulary.size() + " words");
    }
    
    private void addWordToVocabulary(String word) {
        if (!vocabulary.containsKey(word)) {
            vocabulary.put(word, wordList.size());
            wordList.add(word);
        }
    }
    
    public String preprocessText(String text) {
        if (text == null || text.trim().isEmpty()) {
            return "";
        }
        
        // Convert to lowercase
        String processed = text.toLowerCase();
        
        // Remove special characters
        processed = SPECIAL_CHARS.matcher(processed).replaceAll(" ");
        
        // Remove multiple spaces
        processed = MULTIPLE_SPACES.matcher(processed).replaceAll(" ").trim();
        
        return processed;
    }
    
    public List<String> tokenize(String text) {
        String processed = preprocessText(text);
        if (processed.isEmpty()) {
            return new ArrayList<>();
        }
        return Arrays.asList(processed.split("\\s+"));
    }
    
    public double[] textToVector(String text) {
        List<String> tokens = tokenize(text);
        double[] vector = new double[vocabulary.size()];
        
        for (String token : tokens) {
            Integer index = vocabulary.get(token);
            if (index != null) {
                vector[index] = 1.0; // Binary Bag of Words
            }
        }
        
        return vector;
    }
    
    public double[] textToVectorWithTF(String text) {
        List<String> tokens = tokenize(text);
        double[] vector = new double[vocabulary.size()];
        Map<String, Integer> termFrequency = new HashMap<>();
        
        // Count term frequency
        for (String token : tokens) {
            termFrequency.put(token, termFrequency.getOrDefault(token, 0) + 1);
        }
        
        // Create TF vector
        for (Map.Entry<String, Integer> entry : termFrequency.entrySet()) {
            Integer index = vocabulary.get(entry.getKey());
            if (index != null) {
                vector[index] = entry.getValue() / (double) tokens.size();
            }
        }
        
        return vector;
    }
    
    public int getVocabularySize() {
        return vocabulary.size();
    }
    
    public Map<String, Integer> getVocabulary() {
        return new HashMap<>(vocabulary);
    }
}