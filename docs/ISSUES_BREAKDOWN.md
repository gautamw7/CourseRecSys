# Issues Breakdown (GitHub Implementation Plan)

Use this to create issues in GitHub, one at a time. Link each to the Phase, and track progress.

---

## PHASE 0: Foundation Setup (Week 1)

### Issue 0.1: [GitHub Setup] Configure repo with CI/CD
**Labels:** `setup`, `devops`
**Effort:** 2 hours
**Description:**
Set up GitHub repo for collaboration.

**Tasks:**
- [ ] Create `.github/workflows/ci.yml` (pytest, linting)
- [ ] Add `CODEOWNERS` file
- [ ] Set branch protection on `main` (require PR review)
- [ ] Document in SETUP.md

**Definition of Done:**
- Workflow runs on PR
- Tests pass before merge
- All documentation updated

---

### Issue 0.2: [Backend] Initialize FastAPI project skeleton
**Labels:** `backend`, `setup`
**Effort:** 3 hours
**Description:**
Create backend structure with basic routes and tests.

**Tasks:**
- [ ] Create `backend/` directory
- [ ] Set up `app/`, `tests/` directories
- [ ] Write `app/main.py` with FastAPI app + health check
- [ ] Write `app/config.py` for environment variables
- [ ] Write `requirements.txt`
- [ ] Write `tests/conftest.py` (pytest fixtures)
- [ ] Add `.env.example`

**Definition of Done:**
- `uvicorn app.main:app --reload` works
- `GET /health` returns `{"status": "ok"}`
- `pytest` runs (0 tests, all pass)

---

### Issue 0.3: [Frontend] Initialize React project skeleton
**Labels:** `frontend`, `setup`
**Effort:** 2 hours
**Description:**
Create React frontend with basic page structure.

**Tasks:**
- [ ] Create `frontend/` directory (via Create React App or Vite)
- [ ] Create `src/components/`, `src/pages/`, `src/services/`
- [ ] Add `.env.example` (API_BASE_URL)
- [ ] Add `src/services/api.ts` (HTTP client stub)

**Definition of Done:**
- `npm start` runs on `localhost:3000`
- App renders without errors
- Page structure matches wireframe

---

### Issue 0.4: [Docs] Write ADR-001: Vector DB Choice
**Labels:** `docs`, `architecture`
**Effort:** 1 hour
**Description:**
Document why we chose Qdrant over FAISS/Weaviate/Pinecone.

**Tasks:**
- [ ] Create `docs/adr/001-vector-db-choice.md`
- [ ] Explain decision tree and tradeoffs
- [ ] Link from ARCHITECTURE.md

**Definition of Done:**
- ADR is clear, stakeholders agree
- Decision recorded for v2

---

### Issue 0.5: [Docs] Write SETUP.md (Development Guide)
**Labels:** `docs`
**Effort:** 2 hours
**Description:**
Clear instructions for new developers to set up local environment.

**Tasks:**
- [ ] Backend setup (venv, requirements, env vars)
- [ ] Frontend setup (npm install, env vars)
- [ ] Running tests
- [ ] Running dev servers
- [ ] Debugging tips
- [ ] Docker setup (optional for Phase 3)

**Definition of Done:**
- Any developer can follow SETUP.md and run locally

---

### Issue 0.6: [Backend] Add logging & error handling middleware
**Labels:** `backend`, `quality`
**Effort:** 2 hours
**Description:**
Set up structured logging and centralized error handling.

**Tasks:**
- [ ] Configure Python logging (INFO level in dev, ERROR in prod)
- [ ] Add request/response logging middleware
- [ ] Add exception handler (catch all, log, return JSON)
- [ ] Test with sample endpoint

**Definition of Done:**
- Logs appear in console when routes are hit
- Errors return consistent JSON response

---

## PHASE 1: Backend Core (Week 2-3)

### Issue 1.1: [Backend] PDF Extraction Service
**Labels:** `backend`, `feature`
**Effort:** 4 hours
**Depends on:** 0.2
**Description:**
Build PDF parsing service using pdfplumber.

**Tasks:**
- [ ] Create `app/services/pdf_extractor.py`
- [ ] Write `extract_text(file: UploadFile) -> str` function
- [ ] Handle errors (corrupt PDF, not a PDF, etc.)
- [ ] Write unit tests (`tests/unit/test_pdf_extractor.py`)
- [ ] Document with docstring

**Definition of Done:**
- Can extract text from valid PDFs
- Graceful error on invalid files
- 80%+ test coverage

**Test case:**
```python
def test_extract_valid_pdf():
    # Load sample resume
    # Assert text extracted
    # Assert keywords present

def test_extract_corrupt_pdf():
    # Assert raises ValueError
```

---

### Issue 1.2: [Backend] Gemini Resume Extraction
**Labels:** `backend`, `feature`, `llm`
**Effort:** 6 hours
**Depends on:** 1.1, 0.2
**Description:**
Call Gemini API to extract structured profile from resume text.

**Tasks:**
- [ ] Create `app/services/llm_service.py`
- [ ] Write prompt template for resume extraction
- [ ] Write `extract_profile(resume_text: str) -> ResumeProfile` (async)
- [ ] Add retry logic (tenacity)
- [ ] Write unit tests (mock Gemini API)
- [ ] Test with real resume + real Gemini API

**Definition of Done:**
- Gemini extracts JSON with all required fields
- Handles rate limits gracefully
- Returns validated ResumeProfile
- Tests pass (mocked)

**Test cases:**
```python
def test_extract_profile_valid_resume():
    # Mock Gemini response
    # Assert ResumeProfile has skills, goal, etc.

def test_extract_profile_rate_limit():
    # Mock rate limit error
    # Assert retry happens
    # Assert eventual success

def test_extract_profile_invalid_response():
    # Mock bad JSON from Gemini
    # Assert raises ValueError
```

---

### Issue 1.3: [Backend] Qdrant Integration
**Labels:** `backend`, `feature`, `vector-db`
**Effort:** 5 hours
**Depends on:** 0.2
**Description:**
Connect to Qdrant, load course embeddings, implement search.

**Tasks:**
- [ ] Create `app/services/vector_search.py`
- [ ] Connect to Qdrant (local Docker or cloud)
- [ ] Load precomputed embeddings into collection (one-time setup)
- [ ] Write `search(profile: ResumeProfile, limit: int) -> List[Course]`
- [ ] Write unit tests (mock Qdrant or use local)

**Definition of Done:**
- Can query Qdrant collection
- Returns top-N courses with metadata
- Handles connection errors

**Test cases:**
```python
def test_search_returns_top_n():
    # Query with sample embedding
    # Assert returns N courses

def test_search_handles_connection_error():
    # Mock connection failure
    # Assert raises exception with helpful message
```

---

### Issue 1.4: [Backend] Embedding Service (SentenceTransformer)
**Labels:** `backend`, `feature`
**Effort:** 3 hours
**Depends on:** 1.3
**Description:**
Encode user profiles to vectors for search.

**Tasks:**
- [ ] Create `app/services/vector_service.py`
- [ ] Load SentenceTransformer model (cache in memory)
- [ ] Write `encode_profile(profile: ResumeProfile) -> List[float]`
- [ ] Build semantic string from profile
- [ ] Write unit tests

**Definition of Done:**
- Encodes profile to 384-dim vector
- Vector consistent across calls
- Tests pass

---

### Issue 1.5: [Backend] Claude Ranking & Explanations
**Labels:** `backend`, `feature`, `llm`
**Effort:** 6 hours
**Depends on:** 1.4, 0.2
**Description:**
Rank top courses and generate explanations using Claude.

**Tasks:**
- [ ] Create `app/services/recommendation.py`
- [ ] Write prompt template for ranking
- [ ] Write `rank_and_explain(profile, courses) -> List[Recommendation]` (async)
- [ ] Parse Claude response into Recommendations
- [ ] Write unit tests (mock Claude)
- [ ] Test with real Claude API

**Definition of Done:**
- Claude returns ranked courses with explanations
- Explanations reference user's skills/goals
- Tests pass

---

### Issue 1.6: [Backend] POST /upload Endpoint
**Labels:** `backend`, `feature`, `api`
**Effort:** 4 hours
**Depends on:** 1.1, 1.2, 1.4, 1.5
**Description:**
Wire all services into upload endpoint.

**Tasks:**
- [ ] Create `app/routes/upload.py`
- [ ] Write `POST /upload` handler
- [ ] Orchestrate: extract PDF → extract profile → search → rank
- [ ] Return `{ user_profile, recommendations }`
- [ ] Write integration tests

**Definition of Done:**
- End-to-end upload works
- Returns expected response shape
- Error handling works

---

### Issue 1.7: [Backend] POST /questionnaire Endpoint
**Labels:** `backend`, `feature`, `api`
**Effort:** 3 hours
**Depends on:** 1.2, 1.4, 1.5
**Description:**
Alternative path: questionnaire form instead of resume.

**Tasks:**
- [ ] Create `app/routes/recommend.py`
- [ ] Write `POST /questionnaire` handler
- [ ] Map form inputs (goal, topic, level, time) to ResumeProfile
- [ ] Call search → rank
- [ ] Return same response as `/upload`

**Definition of Done:**
- Questionnaire route works
- Same output shape as `/upload`
- Tests pass

---

### Issue 1.8: [Backend] Pydantic Models
**Labels:** `backend`, `quality`
**Effort:** 2 hours
**Depends on:** 1.1
**Description:**
Define request/response schemas.

**Tasks:**
- [ ] Create `app/models/user_profile.py` (ResumeProfile, QuestionnaireInput)
- [ ] Create `app/models/recommendation.py` (Recommendation, Course)
- [ ] Add validation (field constraints, enums)
- [ ] Document with docstrings

**Definition of Done:**
- All schemas defined
- Validation works as expected
- OpenAPI docs auto-generated

---

### Issue 1.9: [Backend] Error Handling & Validation Tests
**Labels:** `backend`, `quality`, `test`
**Effort:** 3 hours
**Depends on:** 1.6, 1.7
**Description:**
Test edge cases and error scenarios.

**Tasks:**
- [ ] Test invalid PDF (corrupted, wrong format)
- [ ] Test Gemini timeout / rate limit
- [ ] Test Qdrant connection failure
- [ ] Test malformed questionnaire input
- [ ] Test empty results (no courses found)

**Definition of Done:**
- All error cases handled gracefully
- User-facing error messages clear
- Logs helpful for debugging

---

## PHASE 2: Frontend (Week 3-4)

### Issue 2.1: [Frontend] Landing Page
**Labels:** `frontend`, `feature`, `ui`
**Effort:** 3 hours
**Depends on:** 0.3
**Description:**
Build landing page with "Choose input method" choice.

**Tasks:**
- [ ] Create `src/pages/Home.tsx`
- [ ] Two buttons: "Upload Resume" | "Answer Questions"
- [ ] Add branding / intro text
- [ ] Add loading states

**Definition of Done:**
- Page renders
- Buttons navigate correctly
- Looks professional

---

### Issue 2.2: [Frontend] Resume Upload Form
**Labels:** `frontend`, `feature`, `ui`
**Effort:** 3 hours
**Depends on:** 2.1, 1.6
**Description:**
File upload form with progress indicator.

**Tasks:**
- [ ] Create `src/components/ResumeUpload.tsx`
- [ ] Drag-and-drop or file picker
- [ ] Show loading spinner while uploading
- [ ] Handle errors (file too large, not PDF)
- [ ] Call `POST /upload` API

**Definition of Done:**
- Can select & upload PDF
- Shows loading state
- Navigates to results on success

---

### Issue 2.3: [Frontend] Questionnaire Form
**Labels:** `frontend`, `feature`, `ui`
**Effort:** 4 hours
**Depends on:** 2.1, 1.7
**Description:**
4-question form (goal, topic, level, time).

**Tasks:**
- [ ] Create `src/components/Questionnaire.tsx`
- [ ] Goal: dropdown (career-switch, upskilling, exploration)
- [ ] Topic: text input (e.g., "Machine Learning")
- [ ] Level: radio buttons (beginner, intermediate, advanced)
- [ ] Time: slider (1-20 hrs/week)
- [ ] Call `POST /questionnaire` API
- [ ] Show loading state

**Definition of Done:**
- Form renders and validates
- API call on submit
- Navigates to results

---

### Issue 2.4: [Frontend] Recommendation Card Component
**Labels:** `frontend`, `feature`, `ui`
**Effort:** 3 hours
**Depends on:** 0.3
**Description:**
Reusable card component for each recommendation.

**Tasks:**
- [ ] Create `src/components/RecommendationCard.tsx`
- [ ] Show: title, platform, difficulty, score %, explanation, link
- [ ] Styling (card layout, colors, hover effects)
- [ ] Link opens course in new tab

**Definition of Done:**
- Card displays all info clearly
- Link works
- Responsive on mobile

---

### Issue 2.5: [Frontend] Results Page
**Labels:** `frontend`, `feature`, `ui`
**Effort:** 4 hours
**Depends on:** 2.4, 1.6, 1.7
**Description:**
Display recommendations after API call.

**Tasks:**
- [ ] Create `src/pages/Results.tsx`
- [ ] Map recommendations to RecommendationCard components
- [ ] Show user profile (optional collapsible)
- [ ] "Try again" button to reset
- [ ] Error state if API fails

**Definition of Done:**
- Page renders recommendations
- All cards visible
- Can retry

---

### Issue 2.6: [Frontend] API Client Service
**Labels:** `frontend`, `feature`
**Effort:** 2 hours
**Depends on:** 0.3
**Description:**
HTTP client for backend communication.

**Tasks:**
- [ ] Create `src/services/api.ts`
- [ ] `uploadResume(file: File)` function
- [ ] `submitQuestionnaire(data: FormData)` function
- [ ] Error handling + retry logic
- [ ] TypeScript types for request/response

**Definition of Done:**
- API calls work
- Errors propagate to UI
- Types match backend (Pydantic models)

---

### Issue 2.7: [Frontend] Loading & Error States
**Labels:** `frontend`, `feature`, `ux`
**Effort:** 2 hours
**Depends on:** 2.2, 2.3, 2.5
**Description:**
Skeleton screens and error messages.

**Tasks:**
- [ ] Create `src/components/LoadingState.tsx`
- [ ] Show while waiting for API
- [ ] Error toast/modal on failure
- [ ] Clear messaging (what went wrong, what to do)

**Definition of Done:**
- UX is smooth (not jarring state changes)
- Errors are helpful

---

## PHASE 3: Polish & Deploy (Week 4-5)

### Issue 3.1: [DevOps] Docker Setup
**Labels:** `devops`, `deployment`
**Effort:** 3 hours
**Depends on:** All backend + frontend done
**Description:**
Containerize backend and frontend.

**Tasks:**
- [ ] Create `backend/Dockerfile`
- [ ] Create `frontend/Dockerfile`
- [ ] Create `docker-compose.yml` (backend + frontend + Qdrant)
- [ ] Document in SETUP.md

**Definition of Done:**
- `docker-compose up` runs full stack
- Can access app at `localhost:3000`

---

### Issue 3.2: [Testing] End-to-End Tests
**Labels:** `test`, `qa`
**Effort:** 4 hours
**Depends on:** All features done
**Description:**
Full flow testing (Cypress or Playwright).

**Tasks:**
- [ ] Create e2e test: upload resume → see results
- [ ] Create e2e test: questionnaire → see results
- [ ] Error scenarios

**Definition of Done:**
- e2e tests pass
- Coverage of happy path + errors

---

### Issue 3.3: [Performance] Load Testing & Optimization
**Labels:** `performance`, `test`
**Effort:** 3 hours
**Depends on:** 3.2
**Description:**
Ensure system handles load; optimize if needed.

**Tasks:**
- [ ] Load test `/upload` endpoint (k6 or Apache JMeter)
- [ ] Profile slow paths
- [ ] Optimize if needed (caching, batching, etc.)

**Definition of Done:**
- System handles 10 concurrent requests
- No timeouts

---

### Issue 3.4: [Docs] API Documentation (OpenAPI)
**Labels:** `docs`
**Effort:** 1 hour
**Depends on:** All API endpoints done
**Description:**
Auto-generated docs from FastAPI.

**Tasks:**
- [ ] Ensure docstrings on all endpoints
- [ ] Verify `/docs` page loads
- [ ] Document response codes (200, 400, 500)

**Definition of Done:**
- FastAPI `/docs` has clear documentation
- Can test endpoints from UI

---

### Issue 3.5: [Deployment] Deploy to Cloud (Render/Railway/Heroku)
**Labels:** `devops`, `deployment`
**Effort:** 4 hours
**Depends on:** 3.1
**Description:**
Deploy to public URL.

**Tasks:**
- [ ] Choose host (Render, Railway, Heroku)
- [ ] Set up CI/CD to auto-deploy on `main` merge
- [ ] Configure environment variables (API keys, URLs)
- [ ] Test deployed app

**Definition of Done:**
- App live at public URL
- Automatic deploys on push

---

### Issue 3.6: [Monitoring] Error Tracking & Logging
**Labels:** `devops`, `quality`
**Effort:** 2 hours
**Depends on:** 3.5
**Description:**
Set up error monitoring (Sentry or similar).

**Tasks:**
- [ ] Add Sentry (or similar) to backend
- [ ] Capture errors, send to dashboard
- [ ] View errors in Sentry UI

**Definition of Done:**
- Errors logged and visible
- Can debug production issues

---

## PHASE 4: v1.1 (Future)

### Issue 4.1: [Feature] User Survey for Feedback
**Labels:** `feature`, `ux`
**Description:**
Post-recommendation survey to measure satisfaction.

---

### Issue 4.2: [Feature] Rate Limiting & Caching
**Labels:** `feature`, `performance`
**Description:**
Cache common queries; rate limit free tier.

---

### Issue 4.3: [Feature] Conversational Questionnaire
**Labels:** `feature`, `llm`
**Description:**
Multi-turn chat instead of 4-question form.

---

---

## How to Use This Breakdown

1. **Create GitHub issues** from each Issue section
2. **Assign to yourself** when starting
3. **Link to PRs** when implementing
4. **Close when Definition of Done is met**
5. **Update status** in project board

### Priority Order
1. **0.1 - 0.6**: Foundation (required to start)
2. **1.1 - 1.9**: Backend core (required for v1)
3. **2.1 - 2.7**: Frontend (required for v1)
4. **3.1 - 3.6**: Polish & deploy (required for launch)

### Time Estimate
- Phase 0: 12 hours
- Phase 1: 32 hours
- Phase 2: 24 hours
- Phase 3: 17 hours
- **Total: ~85 hours (~3-4 weeks @ 20hrs/week)**

