# Domain Language & Ubiquitous Terms

This document defines the shared language for the Course Recommendation System. Use these terms consistently across code, PRs, docs, and conversations.

---

## Core Entities

### **Resume Profile**
The structured representation of a user derived from either a resume PDF or questionnaire responses.

**Fields:**
```json
{
  "skills": ["Python", "SQL", "TensorFlow"],
  "domains_of_interest": ["ML", "Data Science"],
  "education_level": "Bachelor's",
  "experience_years": 0,
  "learning_goal": "career-switch | skill-deepening | exploration",
  "time_per_week_hours": 10,
  "preferred_format": "video | text | mixed",
  "source": "resume | questionnaire"
}
```

**Why condensed:** Extracted from raw resume text via Gemini, removes noise while preserving signal for recommendations. Never used directly for filtering.

---

### **Course Document**
A row from the aggregated course catalog (Coursera, NPTEL, etc.). Lives in Qdrant as a vector payload.

**Core fields:**
```json
{
  "title": "Machine Learning with Python",
  "platform": "coursera",
  "provider": "Stanford Online",
  "description": "...",
  "skills": ["Python", "ML"],
  "difficulty": "intermediate",
  "rating": 4.7,
  "url": "https://..."
}
```

**Why:** Consistent schema across data sources; indexed for search.

---

### **Recommendation**
A Course Document matched to a Resume Profile, ranked by relevance and paired with an AI-generated explanation.

**Structure:**
```json
{
  "course": { ...Course Document... },
  "similarity_score": 0.87,
  "score_percentage": "87%",
  "explanation": "Your Python background and ML interests align well with this course's focus on practical algorithms."
}
```

**Why:** Explanation (not just score) is the value-add; score alone is a Google search.

---

## Processes

### **1. User Onboarding**
User enters the system. Two paths:

- **Path A (Resume)**: User uploads PDF → System extracts text → Gemini parses structured profile
- **Path B (Questionnaire)**: User answers 4 questions → System creates profile JSON directly

**Outcome:** Resume Profile ready for search.

---

### **2. Profile-to-Vector Encoding**
Resume Profile → embedding.

**Steps:**
1. Build a semantic string combining skills, goals, interests, education
2. Pass to SentenceTransformer (all-MiniLM-L6-v2)
3. Get 384-dim vector

**Why:** Allows semantic similarity search, not just keyword matching.

---

### **3. Vector Search**
Query Qdrant with user embedding, retrieve top-N matching Course Documents.

**Parameters:**
- Query vector: 384 dimensions (from Resume Profile)
- Collection: "courses"
- Limit: typically 10-15 results (before filtering)
- Metric: cosine distance (normalized in Qdrant)

**Outcome:** Ranked list of Course Documents + similarity scores.

---

### **4. Ranking + Explanation**
Take top-N search results, send to Claude for re-ranking and explanation generation.

**Claude's task:**
- Read user profile + top courses
- Rank by fit (considering goal, timeline, difficulty)
- Generate 1-sentence explanation per course: "Why this matches you"

**Outcome:** Top-5 Recommendations with explanations.

---

## Terminology Decisions

### **"Recommendation" vs "Match"**
**Decision: "Recommendation"**
- A "match" is what the vector DB returns (similarity score)
- A "recommendation" includes the explanation layer (why it's good for *you*)

---

### **"Query" vs "Search"**
**Decision: "Search"** (when talking to Qdrant), **"Recommendation Request"** (user-facing)
- "Search" = technical vector DB query
- "Recommendation Request" = user asking for suggestions

---

### **"Condensed" vs "Extracted"**
**Decision: "Extracted Resume Profile"**
- "Extracted" = pulling data from source (resume PDF)
- "Condensed" was vague; avoid it

---

### **"Vector Database" vs "Index"**
**Decision: "Vector Database"** (Qdrant), **"Index"** (if referring to FAISS locally)
- Qdrant is a full DB (persistent, queryable)
- FAISS is an index (in-memory, faster but transient)

---

## User Types

### **Student / Early Career**
- ~0-2 years experience
- Learning goal: skill exploration or career preparation
- Data: Resume optional, questionnaire sufficient

### **Career Switcher**
- 5+ years in different field
- Learning goal: specific skill gap closure
- Data: Resume important (shows non-tech background)

### **Upskilling Professional**
- 3+ years in tech/data
- Learning goal: deepening in specific domain
- Data: Resume useful for context

---

## Metrics & Success Criteria

### **Relevance**
Does the user find the recommended courses useful?
- Proxy: Do they click the link? Do they enroll?

### **Diversity**
Are recommendations varied (different platforms, difficulty levels)?
- Rule of thumb: Top-5 should span 2-3 difficulty levels

### **Cold-Start Clarity**
Do users understand how to get started (resume vs questionnaire)?
- Success: <5% user confusion about input

---

## API Contracts

### **POST /upload**
**Request:** `{ file: multipart/form-data }`
**Response:**
```json
{
  "status": "success",
  "user_profile": { ...Resume Profile... },
  "recommendations": [ ...Recommendation[] ... ]
}
```

### **POST /questionnaire**
**Request:**
```json
{
  "goal": "string",
  "topic": "string",
  "level": "beginner | intermediate | advanced",
  "time_per_week": integer
}
```
**Response:** Same as `/upload`

---

## Assumptions & Constraints

1. **Resume parsing is lossy.** Gemini extracts best-effort; users can clarify via follow-up (v2 conversational).
2. **Vector similarity ≠ relevance.** Explanation layer (Claude) bridges this gap.
3. **No user accounts (MVP).** All recommendations are stateless.
4. **English-only (v1).** Courses, prompts, UI all English.
5. **Gemini free tier limitations.** Max RPM ~60; batch requests if scaling.

---

## References

- **Vector DB choice** → See `docs/adr/001-vector-db-choice.md`
- **LLM integration** → See backend services (`app/services/llm_service.py`)
- **Data pipeline** → See README "Data Pipeline" section