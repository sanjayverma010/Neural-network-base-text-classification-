package com.sentiment;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;

@SpringBootApplication
public class SentimentAnalysisApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(SentimentAnalysisApplication.class, args);
    }
    
    @EventListener(ApplicationReadyEvent.class)
    public void onApplicationReady() {
        System.out.println("\n" +
            "╔═══════════════════════════════════════════════════════════╗\n" +
            "║     🧠 SENTIMENT ANALYSIS APPLICATION STARTED 🧠          ║\n" +
            "║                                                           ║\n" +
            "║  🌐 Access the application at: http://localhost:8080      ║\n" +
            "║  📝 API Endpoint: http://localhost:8080/api/sentiment     ║\n" +
            "║                                                           ║\n" +
            "║  Neural Network Architecture:                             ║\n" +
            "║  • Input Layer → Hidden Layer 1 (ReLU)                   ║\n" +
            "║  • Hidden Layer 2 (ReLU + Dropout)                       ║\n" +
            "║  • Output Layer (Sigmoid)                                ║\n" +
            "╚═══════════════════════════════════════════════════════════╝\n");
    }
}