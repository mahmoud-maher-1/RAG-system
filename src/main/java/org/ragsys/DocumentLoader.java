package org.ragsys;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.util.List;
import java.util.Scanner;

public class DocumentLoader {
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;

    public DocumentLoader() {
        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        this.embeddingStore = MilvusEmbeddingStore.builder()
                .host("localhost")
                .port(19530)
                .collectionName("documents")
                .dimension(384)
                .build();
    }

    public void processDocument(Path documentPath) {
        StringBuilder text = new StringBuilder();
        try {
            Scanner scanner = new Scanner(new File(documentPath.toString()));
            while (scanner.hasNextLine()) {
                text.append(scanner.nextLine());
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // Load document
        Document document = Document.from(text.toString(), Metadata.metadata("Author","MM"));

        // Split document
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

        // Embed and store
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        embeddingStore.addAll(embeddings, segments);
    }
}