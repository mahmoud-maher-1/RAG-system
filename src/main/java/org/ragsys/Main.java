package org.ragsys;

import java.nio.file.Paths;

public class Main {
    public static void main(String[] args) {
        DocumentLoader processor = new DocumentLoader();
        RagPipeline rag = new RagPipeline();

        try {
            processor.processDocument(Paths.get("path/to/example/text/file/for/embedding"));
            System.out.println("Documents processed and stored");
        } catch (Exception e) {
            System.err.println("Error processing documents: " + e.getMessage());
        }

        // Query example
        String question = "What are the key features of this system?";
        String answer = rag.answer(question);
        System.out.println("Question: " + question);
        System.out.println("Answer: " + answer);
    }
}