package org.ragsys;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;

import java.util.List;
import java.util.stream.Collectors;

public class RagPipeline {
    private final ChatLanguageModel chatModel;
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;

    public RagPipeline() {
        this.chatModel = OllamaChatModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("mistral")
                .temperature(0.7)
                .build();

        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        this.embeddingStore = MilvusEmbeddingStore.builder()
                .host("localhost")
                .port(19530)
                .collectionName("documents")
                .dimension(384)
                .build();
    }

    public String answer(String question) {
        // Embed the question
        Embedding questionEmbedding = embeddingModel.embed(question).content();

        // Retrieve relevant documents
        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore
                .findRelevant(questionEmbedding, 3);

        // Format context
        String context = relevant.stream()
                .map(match -> match.embedded().text())
                .collect(Collectors.joining("\n\n"));

        // Build prompt with context
        String prompt = String.format(
                "Based on the following context, answer the question:\n\n" +
                        "Context:\n%s\n\n" +
                        "Question: %s\n" +
                        "Answer:",
                context, question
        );

        // Generate response
        String response = chatModel.generate(prompt);
        return response;
    }
}