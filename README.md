<div align="center">

# EdTech Personal Tutor Agent 

Built with **FastAPI, LangChain, Qdrant, Redis, OpenAI**.

Agent provides **personalized tutoring**, **RAG-based explanations**, **adaptive quizzes**, and **student progress tracking** with production-level safeguards.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


</div>

🔍 What This Agent Does

This AI agent acts as a **personal tutor**:

- Explains complex topics step-by-step
- Adapts explanations to student level
- Retrieves answers from course materials (RAG)
- Generates quizzes and tracks progress
- Safely executes Python code examples
- Enforces rate limits, PII protection, and cost control

### Production Features
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Monitoring**: Prometheus metrics + LangSmith traces
- **Security Guards**: PII detection, prompt injection protection
- **Rate Limiting**: Per-user request and token limits (Redis-based)
- **Input Validation**: Sanitization and validation of all inputs
- **Observability**: Structured JSON logging with request tracing
- **Docker Ready**: Multi-stage builds, docker-compose setup
- **Tested**: Unit + Integration tests with >80% coverage
---

##  Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                             │
│  /chat, /quiz, /progress, /health, /metrics                     │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│                    Agent Orchestration                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Tutor Chain  │  │ Memory Mgr   │  │ Rate Limiter │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│                      Core Components                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ RAG Pipeline │  │ LLM Provider │  │ Tools        │          │
│  │ (Multi-Query)│  │ (Fallback)   │  │ (Quiz/Code)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│                    Infrastructure                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Qdrant       │  │ Redis        │  │ Prometheus   │          │
│  │ (Vectors)    │  │ (Cache/Rate) │  │ (Metrics)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

- **API**: FastAPI 0.109+
- **Framework**: LangChain 0.1+
- **Vector Database**: Qdrant 1.7+
- **Cache/Rate Limiting**: Redis 7+
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4 (primary), GPT-3.5 (fallback)
- **Monitoring**: Prometheus, LangSmith, Sentry
- **Logging**: Structlog (JSON format)
- **Deployment**: Docker, Docker Compose

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/edtech-tutor-agent.git
cd edtech-tutor-agent
```

### 2. Setup Environment
```bash
# Copy environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

**Required environment variables:**
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Start with Docker Compose
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f edtech-tutor
```

Services will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/api/v1/docs
- **Qdrant**: http://localhost:6333/dashboard
- **Metrics**: http://localhost:8000/metrics

### 4. Initialize Data (First Time Only)
```bash
# Seed Qdrant with sample course data
docker-compose exec edtech-tutor python scripts/seed_qdrant.py
```

### 5. Test the API
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "student_123",
    "message": "Explain Python recursion to me"
  }'
```

---

### Core Endpoints

#### 1. Chat with Tutor
```http
POST /api/v1/chat
Content-Type: application/json

{
  "student_id": "student_123",
  "message": "What is recursion?",
  "session_id": "session_abc"  // optional
}
```

**Response:**
```json
{
  "response": "Recursion is a programming technique...",
  "session_id": "session_abc",
  "sources": [
    {
      "content": "Relevant course material...",
      "metadata": {"topic": "recursion", "score": 0.92}
    }
  ],
  "tool_results": {},
  "student_progress": {
    "completed_topics": ["variables", "loops"],
    "total_questions": 15,
    "quiz_pass_rate": 0.85
  }
}
```

#### 2. Generate Quiz
```http
POST /api/v1/quiz/generate
Content-Type: application/json

{
  "student_id": "student_123",
  "topic": "Python recursion",
  "count": 5,
  "difficulty": "medium",
  "use_rag": true
}
```

#### 3. Get Student Progress
```http
GET /api/v1/progress/student_123
```

#### 4. Health Check
```http
GET /api/v1/health/ready
```

### Rate Limits

- **Requests**: 100 per hour per student
- **Tokens**: 10,000 per day per student

Rate limit headers are included in responses:
```
X-RateLimit-Remaining-Requests: 95
X-RateLimit-Remaining-Tokens: 9500
```

---

## Monitoring

### Prometheus Metrics

Access metrics at: http://localhost:8000/metrics

**Key Metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `llm_call_duration_seconds` - LLM call latency
- `llm_tokens_used_total` - Token usage
- `qdrant_query_duration_seconds` - Vector search latency
- `student_questions_total` - Questions by topic
- `quiz_completion_rate` - Quiz pass rates
- `rate_limit_exceeded_total` - Rate limit violations

### LangSmith Traces

Enable in `.env`:
```bash
LANGSMITH_ENABLED=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=edtech-tutor
```

View traces at: https://smith.langchain.com/

### Grafana Dashboards

Start monitoring stack:
```bash
docker-compose --profile monitoring up -d
```

Access Grafana: http://localhost:3000 (admin/admin)

### Logs
```bash
# View real-time logs
docker-compose logs -f edtech-tutor

# JSON logs for production
tail -f logs/app.log | jq
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Qdrant Connection Error**
```
QdrantConnectionError: Failed to connect to Qdrant
```

**Solution:**
- Ensure Qdrant is running: `docker-compose ps qdrant`
- Check URL in `.env`: `QDRANT_URL=http://localhost:6333`
- Verify network: `docker network ls`

**2. Rate Limit Exceeded**
```
429 Too Many Requests: rate_limit_exceeded
```

**Solution:**
- Check usage: `GET /api/v1/progress/{student_id}/usage`
- Adjust limits in `.env` if needed
- Wait for window to reset (1 hour for requests, 24h for tokens)

**3. OpenAI API Error**
```
LLMProviderError: Failed to generate completion
```

**Solution:**
- Verify API key: `echo $OPENAI_API_KEY`
- Check OpenAI status: https://status.openai.com/
- Review rate limits on OpenAI dashboard
- Fallback model will activate automatically

**4. Memory/Performance Issues**
```bash
# Check container stats
docker stats edtech-tutor

# Increase resources in docker-compose.yml
services:
  edtech-tutor:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
```

### Debug Mode

Enable detailed logging:
```bash
LOG_LEVEL=DEBUG
LOG_FORMAT=console
```
---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---
<div align="center">

## 👩‍💻 Автор

**Елизавета Кевбрина**

*LLM Engineer • Workflow Automation • AI Integrations*

[![Email](https://img.shields.io/badge/Email-elisa.kevbrina%40yandex.ru-red?style=flat-square&logo=gmail)](mailto:elisa.kevbrina@yandex.ru)
[![GitHub](https://img.shields.io/badge/GitHub-%40LizaKevbrina-black?style=flat-square&logo=github)](https://github.com/LizaKevbrina)

---

**⭐ Star this repo if you find it useful!**

*Made with ❤️ for the AI community*

</div>
