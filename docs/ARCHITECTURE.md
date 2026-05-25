# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React)                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ Landing Page     │  │ Resume Upload    │  │ Results      │  │
│  │ (choose input)   │→ │ Form or Q-form   │→ │ Cards        │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ HTTP POST /upload or /questionnaire
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Route Handler (routes/upload.py or routes/recommend.py)  │  │
│  │  - Validate input                                        │  │
│  │  - Call appropriate service                             │  │
│  │  - Return response                                      │  │
│  └───────────────────────┬──────────────────────────────────┘  │
│                          │                                       │
│        ┌─────────────────┴──────────────────┐                  │
│        ↓                                    ↓                   │
│  ┌──────────────────┐          ┌──────────────────────────┐    │
│  │ PDF Service      │          │ Questionnaire Service    │    │
│  │  - Extract text  │          │  - Validate answers      │    │
│  │  - Parse errors  │          │  - Map to schema         │    │
│  └────────┬─────────┘          └──────────┬───────────────┘    │
│           │                              │                     │
│           └──────────────┬───────────────┘                     │
│                          ↓                                      │
│                  ┌────────────────────┐                        │
│                  │ LLM Service        │                        │
│                  │  - Call Gemini API │                        │
│                  │  - Extract + rank  │                        │
│                  │  - Parse responses │                        │
│                  └────────┬───────────┘                        │
│                           ↓                                     │
│                  ┌────────────────────┐                        │
│                  │ Vector Service     │                        │
│                  │  - Encode profile  │                        │
│                  │  - Query Qdrant    │                        │
│                  │  - Get top-N       │                        │
│                  └────────┬───────────┘                        │
│                           ↓                                     │
│                  ┌────────────────────┐                        │
│                  │ Recommendation     │                        │
│                  │ Service            │                        │
│                  │  - Rank courses    │                        │
│                  │  - Call Claude     │                        │
│                  │  - Generate explns │                        │
│                  └────────┬───────────┘                        │
│                           ↓                                     │
│                   ┌──────────────┐                             │
│                   │ Response     │                             │
│                   │ Schema       │                             │
│                   └──────────────┘                             │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             │ JSON Response
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              External Services (Cloud APIs)                      │
│  ┌──────────────────────┐  ┌──────────────┐                    │
│  │ Gemini 2.0 Flash     │  │ Qdrant Cloud │                    │
│  │ (extract + ranking)  │  │ (vector db)  │                    │
│  └──────────────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow (Detailed)

### **Path 1: Resume Upload**

```
1. USER uploads PDF
   ↓
2. FRONTEND sends POST /upload { file: PDF }
   ↓
3. BACKEND routes/upload.py
   a. Validate: file is PDF, size < 5MB
   b. Call services/pdf_extractor.extract_text(pdf)
   ↓
4. PDF EXTRACTOR (pdfplumber)
   a. Open PDF
   b. Extract all text
   c. Return raw text
   ↓
5. LLM SERVICE (routes/upload.py calls)
   a. Format prompt: "Extract structured info from resume: {text}"
   b. Call Gemini API
   c. Parse JSON response → Resume Profile
   d. Validate schema (Pydantic)
   ↓
6. VECTOR SERVICE
   a. Build semantic string from Resume Profile
   b. Encode with SentenceTransformer
   c. Return 384-dim vector
   ↓
7. VECTOR SEARCH (Qdrant)
   a. Query: vector similarity
   b. Limit: 10 results
   c. Return: [Course1, Course2, ..., Course10] with scores
   ↓
8. RECOMMENDATION SERVICE
   a. Take top-10 courses + user profile
   b. Call Gemini API: "Rank these courses for this user, explain top-5"
   c. Parse response: [Recommendation1, ..., Recommendation5]
   ↓
9. RESPONSE MODEL
   a. Return: { user_profile, recommendations: [...] }
   ↓
10. FRONTEND receives JSON
    a. Display each recommendation as a card
    b. Show course title, platform, difficulty, score%, explanation, link
```

### **Path 2: Questionnaire**

```
Same as Path 1, but:
- Step 3 SKIPS PDF extraction
- Step 5 is replaced by:
  a. User submits { goal, topic, level, time_per_week }
  b. Map to Resume Profile directly (no LLM call)
  c. Continue from Step 6
```

---

## Service Responsibilities

### **routes/**
**What:** HTTP handlers
**Does:** Validate input, call services, return responses
**Example:**
```python
@router.post("/upload")
async def upload(file: UploadFile):
    text = await pdf_extractor.extract_text(file)
    profile = await llm_service.extract_profile(text)
    courses = await vector_service.search(profile)
    recs = await recommendation_service.rank(profile, courses)
    return {"profile": profile, "recommendations": recs}
```
**Key:** Routes should be thin (max 20 lines each).

---

### **services/pdf_extractor.py**
**What:** Pure functions for PDF handling
**Does:** Extract text, handle errors gracefully
**Example:**
```python
async def extract_text(file: UploadFile) -> str:
    # Validate file
    # Open with pdfplumber
    # Extract all text
    # Return or raise ValueError
    pass
```
**Why:** Testable without HTTP.

---

### **services/llm_service.py**
**What:** Gemini + Claude API wrappers
**Does:** Format prompts, call APIs, parse responses
**Example:**
```python
async def extract_profile(resume_text: str) -> ResumeProfile:
    prompt = EXTRACTION_PROMPT.format(text=resume_text)
    response = await gemini_client.generate(prompt)
    parsed = json.loads(response)
    return ResumeProfile(**parsed)

async def generate_explanations(profile: ResumeProfile, courses: List[Course]) -> str:
    prompt = RANKING_PROMPT.format(profile=profile, courses=courses)
    response = await claude_client.generate(prompt)
    return response
```
**Why:** Centralized API logic; easy to swap providers or add retries.

---

### **services/vector_search.py**
**What:** Qdrant integration
**Does:** Connect to DB, query, return results
**Example:**
```python
async def search(profile: ResumeProfile, limit: int = 10) -> List[Course]:
    # Build semantic string from profile
    vector = encode(semantic_string)
    # Query Qdrant
    results = qdrant_client.query("courses", vector, limit)
    # Parse into Course objects
    courses = [Course(**r.payload) for r in results.points]
    return courses
```
**Why:** Testable; can mock Qdrant for unit tests.

---

### **services/recommendation.py**
**What:** Ranking + explanation logic
**Does:** Orchestrate ranking, call Claude, return Recommendations
**Example:**
```python
async def rank_and_explain(profile: ResumeProfile, courses: List[Course]) -> List[Recommendation]:
    # Take top-5 from courses (already ranked by Qdrant)
    top_5 = courses[:5]
    # Get explanations from Claude
    explanations = await llm_service.generate_explanations(profile, top_5)
    # Parse explanations
    recs = [
        Recommendation(
            course=top_5[i],
            similarity_score=top_5[i].score,
            explanation=explanations[i]
        )
        for i in range(len(top_5))
    ]
    return recs
```
**Why:** Business logic separate from HTTP.

---

### **models/**
**What:** Pydantic schemas for validation
**Does:** Define request/response shapes
**Example:**
```python
class ResumeProfile(BaseModel):
    skills: List[str]
    domains_of_interest: List[str]
    education_level: str
    experience_years: int
    learning_goal: str  # "career-switch" | "upskilling" | "exploration"
    time_per_week_hours: int
    source: str  # "resume" | "questionnaire"
```
**Why:** Type safety; auto-validation; OpenAPI docs.

---

## Dependencies & Error Handling

### **API Key Management**
```python
# config.py
from pydantic import SecretStr
from functools import lru_cache

class Settings(BaseSettings):
    GEMINI_API_KEY: SecretStr
    CLAUDE_API_KEY: SecretStr
    QDRANT_URL: str
    QDRANT_API_KEY: SecretStr
    
    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()
```
**Why:** Centralized, type-safe, loaded from env.

---

### **Error Handling Pattern**
```python
# In routes/
try:
    result = await service.do_something(input)
    return {"status": "success", "result": result}
except ValueError as e:
    logger.error(f"Validation error: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="Internal error")
```
**Why:** Specific error handling; clear user messages; logging for debugging.

---

### **Retry Logic (for LLM calls)**
```python
# In services/llm_service.py
import tenacity

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10)
)
async def extract_profile(resume_text: str) -> ResumeProfile:
    response = await gemini_client.generate(prompt)
    return parse_response(response)
```
**Why:** Gemini rate limits; backoff prevents thundering herd.

---

## Testing Strategy

### **Unit Tests** (services/)
```python
# tests/unit/test_llm_service.py

@pytest.fixture
def sample_resume():
    return "Name: John Doe\nSkills: Python, SQL\n..."

def test_extract_profile_valid_resume(sample_resume):
    profile = extract_profile(sample_resume)
    assert profile.skills is not None
    assert "Python" in profile.skills

def test_extract_profile_invalid_json():
    with pytest.raises(ValueError):
        extract_profile("garbage data")
```

### **Integration Tests** (routes/)
```python
# tests/integration/test_upload_endpoint.py

@pytest.mark.asyncio
async def test_upload_endpoint(client, sample_pdf):
    response = client.post("/upload", files={"file": sample_pdf})
    assert response.status_code == 200
    data = response.json()
    assert "user_profile" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) == 5
```

---

## Performance Considerations

| Operation | Latency | Strategy |
|-----------|---------|----------|
| PDF extraction | 1-2s | Pdfplumber; async |
| Gemini API | 5-10s | Retry + timeout |
| Vector encoding | 500ms | CPU-bound; cache model |
| Qdrant search | <200ms | Cloud DB; indexed |
| Claude API | 5-10s | Parallel with other steps |
| **Total user-to-response** | **15-25s target** | Optimize later if needed |

---

## Deployment Architecture

### **Local Development**
```
uvicorn app.main:app --reload
Qdrant running locally (Docker) or cloud URL
```

### **Production (v1)**
```
Frontend: Static React build → CDN/S3
Backend: FastAPI in Docker → Cloud host
Qdrant: Cloud instance
LLM APIs: Called from backend (API keys in secrets)
```

---

## Decision Log

### **Q: Why Qdrant over FAISS?**
**A:** Qdrant is persistent; can add replicas later. FAISS is in-memory, faster locally but doesn't scale.

### **Q: Why Gemini for both extraction and ranking?**
**A:** Gemini 2.0 Flash free tier is fast and cheap; handles both tasks well. No API costs for personal project.

### **Q: Why SentenceTransformer (not Claude embeddings)?**
**A:** Self-hosted; no API calls needed; fast; proven in prototype.

---

See also: `docs/adr/` for major decisions.
