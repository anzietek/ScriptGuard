# PROMPT FOR ARCHITECTURE ANALYSIS AGENT

## OBJECTIVE
Create a comprehensive technical presentation document (Markdown format) that thoroughly analyzes the ScriptGuard project architecture, module responsibilities, and complete ML training/inference pipeline flow based on:
- Source code in `src/scriptguard/` directory
- Configuration file `config.yaml`
- Project structure and dependencies

## REQUIRED ANALYSIS SCOPE

### 1. PROJECT OVERVIEW & ARCHITECTURE
- **Technology Stack**: Programming languages, frameworks, key libraries
- **Architecture Pattern**: Overall design pattern (pipeline-based, microservices, etc.)
- **Core Components**: High-level system diagram showing main modules and their interactions
- **Data Flow**: How data moves through the system from ingestion to inference

### 2. MODULE RESPONSIBILITIES (Deep Dive)
Analyze each module in `src/scriptguard/` and provide:
- **Purpose**: What problem does this module solve?
- **Key Classes/Functions**: Main interfaces and their signatures
- **Dependencies**: What other modules does it depend on?
- **Configuration**: Relevant config.yaml sections
- **Code Insights**: Notable implementation details, design patterns used

Required modules to analyze:
- `data_sources/` - Data collection and ingestion
- `preprocessing/` - Data preparation and cleaning
- `steps/` - ZenML pipeline steps
- `pipelines/` - Pipeline orchestration
- `models/` - Model architecture and configuration
- `training/` - Training logic and optimization
- `inference/` - Model serving and prediction
- `rag/` - RAG (Retrieval-Augmented Generation) implementation
- `database/` - Data persistence layer
- `monitoring/` - Logging and metrics
- `api/` - REST API endpoints
- `utils/` - Helper functions and utilities

### 3. COMPLETE TRAINING PIPELINE FLOW
Provide a **step-by-step detailed walkthrough** of the training process:

#### 3.1 Data Preparation Phase
- **Data Sources**: Where does training data come from? (GitHub, malware repositories, etc.)
- **Data Collection**: How is data fetched and filtered? (API calls, keywords, rate limiting)
- **Data Quality**: Validation, deduplication, filtering strategies
- **Data Augmentation**: Techniques used to balance malicious/benign samples
- **Data Format**: Structure of training examples (input format, labels, metadata)

#### 3.2 Vector Embedding Phase
- **Embedding Model**: Which model is used for code embeddings?
- **Tokenization**: How is code tokenized before embedding?
- **Chunking Strategy**: How are large scripts split? (hierarchical, sliding window, etc.)
- **Vector Storage**: Where and how are embeddings stored? (Qdrant, dimensions, indexing)
- **Deduplication**: MinHash LSH or other similarity techniques
- **Code Example**: Show actual code snippet demonstrating embedding process

#### 3.3 QLoRA Fine-tuning Phase
- **Base Model**: Which pre-trained model is used? (StarCoder, CodeLlama, etc.)
- **QLoRA Configuration**: 
  - LoRA rank (r), alpha, dropout
  - Quantization settings (4-bit, 8-bit)
  - Target modules (attention layers, MLP, etc.)
- **Training Hyperparameters**:
  - Learning rate, scheduler type
  - Batch size, gradient accumulation
  - Number of epochs, warmup steps
  - Loss function (weighted? focal?)
- **Optimization Techniques**:
  - Flash Attention 2 usage
  - Gradient checkpointing
  - Mixed precision training
- **Hardware Requirements**: GPU memory, CUDA version
- **Code Example**: Show training loop or key configuration

#### 3.4 Evaluation Phase
- **Metrics**: Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **Validation Strategy**: Train/val/test split ratios, cross-validation
- **Threshold Tuning**: How is classification threshold determined?
- **Error Analysis**: How are false positives/negatives analyzed?
- **Model Checkpointing**: Saving best models, early stopping criteria

### 4. INFERENCE & DEPLOYMENT FLOW
Detailed explanation of how the trained model is used for malware detection:

#### 4.1 API Request Flow
- **Input**: Script content arrives via REST API
- **Preprocessing**: Text normalization, length validation
- **RAG Context Retrieval**: 
  - Query vector store (Qdrant) for similar malicious patterns
  - Few-shot prompt construction with retrieved examples
- **Model Inference**:
  - Prompt formatting
  - Constrained decoding (BENIGN/MALICIOUS token forcing)
  - Confidence score calculation from logits
- **Response**: JSON with classification, confidence, reasoning, related CVEs

#### 4.2 Model Loading & Optimization
- **Adapter Loading**: How PEFT/LoRA adapters are loaded
- **Inference Optimization**: Quantization, attention implementation
- **Batching**: Single vs batch inference
- **Caching**: Response caching strategies

### 5. KEY TECHNICAL HIGHLIGHTS
Identify and explain the most important/interesting technical aspects:
- **Novel Approaches**: Unique solutions or architectural choices
- **Performance Optimizations**: Speed/memory improvements
- **Robustness Features**: Error handling, retry logic, graceful degradation
- **Security Considerations**: API authentication, input validation
- **Scalability**: How system handles load, horizontal scaling

### 6. CONFIGURATION DEEP DIVE
Analyze `config.yaml` structure:
- **Sections Breakdown**: Explain each major configuration section
- **Critical Parameters**: Most important settings and their impact
- **Environment-Specific**: Dev vs production configurations
- **Secrets Management**: How API keys and credentials are handled

### 7. MERMAID DIAGRAMS
Include at least 3 visual diagrams:
1. **System Architecture Diagram**: High-level component interaction
2. **Training Pipeline Flowchart**: Step-by-step training process
3. **Inference Request Sequence**: API request → response flow

## OUTPUT FORMAT REQUIREMENTS

Structure the Markdown document as follows:

```markdown
# ScriptGuard: ML-Based Malicious Script Detection System
## Technical Architecture & Implementation Analysis

---

## Table of Contents
1. Executive Summary
2. System Architecture Overview
3. Module-by-Module Analysis
4. Training Pipeline Deep Dive
5. Inference & Deployment Pipeline
6. Technical Highlights & Innovations
7. Configuration Management
8. Performance & Scalability
9. Conclusion & Future Improvements

---

## 1. Executive Summary
[2-3 paragraph overview of the system]

## 2. System Architecture Overview
[Architecture diagram + explanation]

## 3. Module-by-Module Analysis
### 3.1 Data Sources Module
**Location**: `src/scriptguard/data_sources/`
**Purpose**: [Detailed explanation]
**Key Components**:
- `github_source.py`: [Description + code snippets]
- `malwarebazaar_source.py`: [Description]
...

[Continue for all modules]

## 4. Training Pipeline Deep Dive
### 4.1 Data Preparation
[Detailed step-by-step explanation with code references]

### 4.2 Vector Embedding
[Detailed explanation with diagrams]

### 4.3 QLoRA Fine-tuning
[Detailed explanation with hyperparameters table]

### 4.4 Evaluation
[Metrics, validation strategy, results interpretation]

## 5. Inference & Deployment Pipeline
[Detailed flow from API request to response]

## 6. Technical Highlights & Innovations
[Key interesting technical decisions and their rationale]

## 7. Configuration Management
[config.yaml structure and important parameters]

## 8. Performance & Scalability
[Performance characteristics, bottlenecks, scaling strategies]

## 9. Conclusion & Future Improvements
[Summary and potential enhancements]
```

## RESEARCH INSTRUCTIONS FOR AGENT

1. **Read ALL relevant files** in `src/scriptguard/` directory recursively
2. **Parse config.yaml** thoroughly to understand all configuration options
3. **Trace code flow** from data ingestion through training to inference
4. **Extract actual code snippets** to illustrate key concepts (not pseudocode)
5. **Identify configuration-code relationships**: How config values are used in code
6. **Look for comments and docstrings** in code for additional context
7. **Check for patterns**: ZenML steps, PEFT/LoRA usage, RAG implementation
8. **Verify technical details**: Model IDs, hyperparameters, API endpoints

## QUALITY CRITERIA

The resulting document should:
- ✅ Be **comprehensive** (15-25 pages of content)
- ✅ Include **real code snippets** from the project
- ✅ Have **clear visualizations** (Mermaid diagrams)
- ✅ Explain **WHY** decisions were made, not just WHAT
- ✅ Be **technically accurate** based on actual source code
- ✅ Be **presentation-ready** for technical stakeholders
- ✅ Use **professional technical writing** style
- ✅ Include **specific references** to files and line numbers where relevant

---

**Agent, your mission is to create an outstanding technical presentation document that showcases the ScriptGuard system's architecture and implementation in impressive detail. Begin your analysis now!**

