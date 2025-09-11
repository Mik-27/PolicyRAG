# PolicyRAG: AI-Powered Policy Document Retrieval and Question Answering System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-green.svg)](https://kubernetes.io/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## 🔍 Overview

PolicyRAG is an advanced Retrieval-Augmented Generation (RAG) system specifically designed for institutional policy document management and question-answering. It combines state-of-the-art vector search capabilities with local language model generation to provide accurate, contextual answers to policy-related queries without relying on external APIs.

## � System Architecture & Data Flow

### High-Level Architecture

PolicyRAG follows a microservices architecture with five main components working together to provide intelligent policy retrieval and question answering:

1. **Document Collection Service**: Automated web scraping from PowerDMS
2. **Processing Pipeline**: Text extraction, preprocessing, and embedding generation
3. **Vector Database**: Elasticsearch cluster for semantic search
4. **LLM Service**: Local Ollama instance serving Gemma2:2b model
5. **Web Interface**: Real-time Flask application with chat interface

### Data Flow Process

#### Phase 1: Document Ingestion

**Step 1: Web Scraping**

-   The system begins by accessing the PowerDMS portal using Selenium WebDriver
-   ChromeDriver navigates through the document tree structure automatically
-   PDF documents are identified and downloaded to the local storage directory
-   A 5-minute timeout ensures the scraping process doesn't run indefinitely

**Step 2: Text Extraction**

-   PyPDF2 library processes each downloaded PDF file
-   Text content is extracted page by page and combined into a single document
-   The system handles various PDF formats and encoding issues
-   Extracted text undergoes initial validation for content quality

**Step 3: Text Preprocessing**

-   NLTK performs stopword removal using English language corpus
-   WordNet lemmatization reduces words to their root forms
-   Regular expressions clean special characters while preserving important punctuation
-   Text is chunked into manageable segments for embedding generation

#### Phase 2: Vector Embedding Generation

**Step 1: Tokenization**

-   BAAI/bge-large-en-v1.5 tokenizer converts text into tokens
-   Maximum sequence length is limited to 512 tokens per chunk
-   Padding and truncation ensure consistent input dimensions
-   Token attention masks are generated for proper model processing

**Step 2: Embedding Creation**

-   The BGE model processes tokenized text through transformer layers
-   Last token pooling extracts meaningful representations from the final hidden states
-   Each text chunk is converted into a 1024-dimensional dense vector
-   Embeddings capture semantic meaning and context relationships

**Step 3: Quality Validation**

-   System validates embedding dimensions match expected 1024 values
-   Numerical checks ensure all vector components are valid floating-point numbers
-   Duplicate detection prevents redundant document indexing

#### Phase 3: Vector Database Storage

**Step 1: Index Management**

-   Elasticsearch creates a "policy" index with proper vector field mappings
-   HNSW (Hierarchical Navigable Small World) algorithm enables fast approximate nearest neighbor search
-   Cosine similarity is configured as the primary distance metric
-   Index settings optimize for both search speed and accuracy

**Step 2: Document Indexing**

-   Bulk indexing operations efficiently store multiple documents simultaneously
-   Each document contains metadata (ID, PDF path), original text, and embedding vector
-   Elasticsearch automatically creates inverted indices for text search capabilities
-   Document versioning tracks updates and modifications

#### Phase 4: Query Processing & Retrieval

**Step 1: Query Reception**

-   User submits a natural language question through the web chat interface
-   SocketIO establishes real-time bidirectional communication
-   Query preprocessing applies the same text cleaning pipeline used for documents

**Step 2: Query Embedding**

-   User query undergoes identical tokenization and embedding generation process
-   The same BGE model creates a 1024-dimensional query vector
-   Query vector represents the semantic intent and meaning of the user's question

**Step 3: Similarity Search**

-   Elasticsearch performs vector similarity search using cosine distance
-   The system retrieves top-k most relevant documents (typically 5-10)
-   Hybrid search combines vector similarity with traditional keyword matching
-   Results are ranked by relevance scores for optimal context selection

**Step 4: Context Assembly**

-   Retrieved documents are processed through cumulative relevance scoring
-   System concatenates document texts until relevance threshold is reached (typically cumulative score > 4.5)
-   Context window management ensures LLM input stays within token limits
-   Document sources are preserved for transparency and citation

#### Phase 5: Response Generation

**Step 1: Prompt Construction**

-   System creates a structured prompt combining user query with retrieved context
-   Prompt engineering includes role definition and instruction formatting
-   Context documents are clearly delineated with separators
-   Instructions guide the LLM to focus on policy-specific information

**Step 2: LLM Processing**

-   Ollama serves the local Gemma2:2b model for response generation
-   Model processes the combined prompt and context through transformer layers
-   Local deployment ensures data privacy and eliminates external API dependencies
-   Generation parameters control response length and creativity

**Step 3: Response Delivery**

-   Generated response is validated for coherence and relevance
-   SocketIO streams the response back to the user interface in real-time
-   Response time metrics are captured for performance monitoring
-   Complete query-response cycle is logged for audit and improvement

### Microservices Communication

#### Service Orchestration

**Kubernetes Pod Communication**

-   All services communicate through internal Kubernetes DNS resolution
-   Service discovery enables dynamic endpoint resolution across pods
-   Network policies ensure secure inter-service communication
-   Load balancing distributes requests across multiple replicas when scaled

**Secret Management Flow**

-   HashiCorp Vault stores all sensitive credentials and configuration
-   Vault Agent Injector automatically mounts secrets as files in pods
-   Application startup scripts source these secret files as environment variables
-   Secret rotation occurs transparently without application restarts

#### Data Persistence Strategy

**Document Storage**

-   Persistent Volumes store downloaded PDF documents across pod restarts
-   Document scraper jobs populate shared storage accessible by processing pods
-   Volume claims ensure data durability and availability

**Vector Database Persistence**

-   Elasticsearch uses StatefulSets with persistent storage for data durability
-   Index data persists across cluster restarts and node failures
-   Snapshot and restore capabilities enable backup and disaster recovery

**Model Persistence**

-   Ollama model files are stored in persistent volumes
-   Model initialization jobs download required models once per deployment
-   Shared model storage enables multiple LLM service replicas

### Scalability & Performance Optimizations

#### Horizontal Scaling

-   Individual microservices can scale independently based on load
-   Elasticsearch cluster can expand with additional data nodes
-   Multiple application replicas handle increased user traffic
-   Load balancers distribute requests evenly across service instances

#### Caching Strategy

-   Query result caching reduces redundant embedding computations
-   LRU (Least Recently Used) caching for frequently accessed documents
-   Redis integration enables distributed caching across multiple application instances

#### Batch Processing

-   Document processing occurs in configurable batch sizes
-   Bulk Elasticsearch operations improve indexing throughput
-   Parallel processing utilizes multi-core systems efficiently

This comprehensive data flow ensures PolicyRAG delivers accurate, contextual responses while maintaining high performance, security, and scalability for institutional policy management needs.

## 🛠️ Technology Stack

### Backend Technologies

-   **Python 3.12**: Core application development
-   **Flask + SocketIO**: Real-time web application framework
-   **Elasticsearch 8.12**: Vector database and search engine
-   **PyTorch**: Machine learning framework for embeddings
-   **Transformers**: Hugging Face library for NLP models
-   **Ollama**: Local LLM serving with Gemma2:2b model

### Infrastructure & DevOps

-   **Docker**: Containerization with multi-stage builds
-   **Kubernetes**: Container orchestration with StatefulSets and Jobs
-   **HashiCorp Vault**: Secret management and credential rotation
-   **Minikube**: Local Kubernetes development environment

### Data Processing

-   **Selenium WebDriver**: Automated document collection
-   **PyPDF2**: PDF text extraction and processing
-   **NLTK**: Natural language preprocessing (stopwords, lemmatization)
-   **NumPy**: Numerical operations for vector computations

### Frontend

-   **HTML5/CSS3**: Modern web interface
-   **Bootstrap 5**: Responsive UI framework
-   **JavaScript/Socket.IO**: Real-time bidirectional communication

### Benefits Over Traditional Systems

-   **24/7 Availability**: No human intervention required for basic policy queries
-   **Consistent Responses**: Eliminates variations in policy interpretation
-   **Cost Effective**: Reduces helpdesk burden and manual policy research
-   **Audit Trail**: Complete logging of all queries and responses
-   **Privacy Preserving**: Local deployment keeps sensitive data on-premises

## 🔧 Local Development Setup

### Prerequisites

```bash
# Required software
- Python 3.12+
- Docker Desktop
- Git
- Chrome Browser (for web scraping)

# Optional but recommended
- Minikube (for local Kubernetes testing)
- kubectl
- Helm 3.x
```

### Environment Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Mik-27/PolicyRAG.git
    cd PolicyRAG
    ```

2. **Create Virtual Environment**

    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Environment Configuration**
   Create a `.env` file (for local development only):

    ```bash
    # Elasticsearch Configuration
    ELASTIC_PASSWORD=your_elastic_password
    ELASTIC_CLOUD_ID=your_cloud_id  # For Elastic Cloud
    ELASTIC_API_KEY=your_api_key    # For Elastic Cloud

    # Hugging Face (for model downloads)
    HF_ACCESS_TOKEN=your_hf_token

    # Ollama Configuration
    OLLAMA_HOST=http://localhost:11434
    ```

5. **Download Required NLTK Data**
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

### Local Services Setup

1. **Start Elasticsearch**

    ```bash
    # Using Docker
    docker run -d \
      --name elasticsearch \
      -p 9200:9200 \
      -p 9300:9300 \
      -e "discovery.type=single-node" \
      -e "xpack.security.enabled=false" \
      docker.elastic.co/elasticsearch/elasticsearch:8.12.2
    ```

2. **Start Ollama**

    ```bash
    # Install Ollama locally or use Docker
    docker run -d \
      --name ollama \
      -p 11434:11434 \
      ollama/ollama:latest

    # Pull the Gemma model
    docker exec ollama ollama pull gemma2:2b
    ```

3. **Run the Application**

    ```bash
    python app.py
    ```

4. **Access the Interface**
   Navigate to `http://localhost:5000/chat`

## ☸️ Kubernetes Deployment

### Prerequisites

```bash
# Install Minikube for local development
minikube start --cpus=4 --memory=8192MB --driver=docker

# Install Helm for package management
helm repo add hashicorp https://helm.releases.hashicorp.com
helm repo update
```

### HashiCorp Vault Setup

```bash
# Create Vault namespace
kubectl create namespace vault

# Deploy Vault in development mode
helm install vault hashicorp/vault \
  --namespace vault \
  --set "server.dev.enabled=true" \
  --set "injector.enabled=true"

# Configure Vault for Kubernetes authentication
kubectl exec -it vault-0 -n vault -- /bin/sh -c "
  vault login root
  vault auth enable kubernetes
  vault write auth/kubernetes/config \
    kubernetes_host=\"https://kubernetes.default.svc\" \
    token_reviewer_jwt=\"\$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)\" \
    kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
    issuer=\"https://kubernetes.default.svc.cluster.local\"
"

# Create secrets and policies
kubectl exec -it vault-0 -n vault -- /bin/sh -c "
  vault secrets enable -path=policyrag kv-v2
  vault kv put policyrag/config \
    ELASTIC_PASSWORD=\"your_password\" \
    HF_ACCESS_TOKEN=\"your_token\" \
    ELASTIC_API_KEY=\"your_api_key\" \
    ELASTIC_CLOUD_ID=\"your_cloud_id\"

  vault policy write policyrag - <<EOF
path \"policyrag/data/*\" {
  capabilities = [\"read\"]
}
EOF

  vault write auth/kubernetes/role/policyrag \
    bound_service_account_names=policyrag \
    bound_service_account_namespaces=default \
    policies=policyrag \
    ttl=24h
"
```

### Application Deployment

```bash
# Build and load the application image
docker build -t policyrag:latest .
minikube image load policyrag:latest

# Deploy all components
kubectl apply -f k8s/sa.yaml              # Service accounts
kubectl apply -f k8s/configmap.yaml       # Configuration
kubectl apply -f k8s/elasticsearch.yaml   # Vector database
kubectl apply -f k8s/ollama.yaml          # LLM serving
kubectl apply -f k8s/policyrag-pvc-service.yaml  # Storage and networking
kubectl apply -f k8s/model-init-job.yaml  # Model initialization
kubectl apply -f k8s/scrape-vectorize-job.yaml   # Document processing
kubectl apply -f k8s/policyrag-deployment.yaml   # Main application
kubectl apply -f k8s/policyrag-ingress.yaml      # External access

# Monitor deployment
kubectl get pods -w
kubectl logs -f deployment/policyrag
```

### Access the Application

```bash
# Port forwarding for development
kubectl port-forward svc/policyrag 5000:5000

# Or use Minikube service
minikube service policyrag --url

# For production, configure ingress with proper DNS
```

## 🔧 Configuration Management

### Vault Secret Management

The application uses HashiCorp Vault for secure secret management:

```yaml
# Vault agent injector annotations
annotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "policyrag"
    vault.hashicorp.com/agent-inject-secret-config: "policyrag/data/config"
    vault.hashicorp.com/agent-inject-template-config: |
        {{- with secret "policyrag/data/config" -}}
        export ELASTIC_PASSWORD="{{ .Data.data.ELASTIC_PASSWORD }}"
        export HF_ACCESS_TOKEN="{{ .Data.data.HF_ACCESS_TOKEN }}"
        export ELASTIC_API_KEY="{{ .Data.data.ELASTIC_API_KEY }}"
        export ELASTIC_CLOUD_ID="{{ .Data.data.ELASTIC_CLOUD_ID }}"
        {{- end -}}
```

### Environment-Specific Configuration

```bash
# Development
ELASTIC_HOST=http://localhost:9200
OLLAMA_HOST=http://localhost:11434

# Docker Compose
ELASTIC_HOST=http://elasticsearch:9200
OLLAMA_HOST=http://ollama:11434

# Kubernetes
ELASTIC_HOST=http://elasticsearch:9200
OLLAMA_HOST=http://ollama:11434
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

-   Follow PEP 8 style guidelines
-   Add unit tests for new functionality
-   Update documentation for API changes
-   Ensure Docker builds pass
-   Test Kubernetes deployments

## 🙏 Acknowledgments

-   **Elasticsearch** for providing robust search and analytics capabilities
-   **Hugging Face** for the transformers library and pre-trained models
-   **Ollama** for simplified local LLM deployment
-   **HashiCorp** for Vault secret management solutions
-   **BAAI** for the BGE embedding models

**Built with ❤️ for institutional policy management and AI-powered knowledge retrieval**
