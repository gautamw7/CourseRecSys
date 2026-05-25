# CLAUDE.md — Project Context for Claude Code

This file holds session-independent context for collaborative development with Claude Code.

---

## Project Goals

1. **Build a Course Recommendation System** — FastAPI + React web app
2. **Learn production patterns** — Proper docs, git workflow, testing, project management
3. **Maintain backend ownership** — You implement, I mentor; no code takeover
4. **Transfer learning to Syngenta** — FastAPI, modular design, git discipline, LLM integration, testing strategies

---

## Code Philosophy

### **Golden Rules**
1. **Code is for humans first.** Readable > clever. A junior engineer should understand it in isolation.
2. **Modular design mandatory.** Each function/class has one job; easy to test, debug, modify.
3. **No black boxes.** Every design choice is justified (comments only on the "why", not the "what").
4. **Backend is user-owned.** I review, suggest, teach—but you write it. No swapping in generated code without your understanding.
5. **Front-end can be hands-on.** You said you're OK with more collaboration here; we can iterate faster.

### **Documentation is Required**
- Every module has a docstring explaining its role
- API endpoints have request/response examples in code
- Non-obvious logic gets a comment (why, not what)
- PRs must reference issues and explain the change

### **Testing is Not Optional**
- Unit tests for business logic (services/)
- Integration tests for API endpoints
- Run tests before every commit
- Aim for 70%+ coverage on critical paths

---

## Stack Decisions

| Choice | Rationale |
|--------|-----------|
| **FastAPI** | Type hints, async, fast, auto-docs; good foundation for Syngenta patterns |
| **React 18** | Component-based, hooks, TypeScript support; you're learning here, I can help iterate |
| **Pydantic** | Validation + serialization; teaches you data modeling |
| **pytest** | Industry standard; TDD friendly |
| **Qdrant** | You've prototyped with it; cloud-ready |
| **Gemini (extraction) + Claude (ranking)** | Gemini free tier for heavy lifting; Claude for your learning goal |
| **SentenceTransformer** | Lightweight, CPU-friendly, proven in your prototype |

---

## Git Discipline

- **Commits**: Conventional format (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
- **Branches**: `main` (prod) ← `develop` (staging) ← `feat/name` (feature)
- **PRs**: Link to issue, describe change, show before/after for UI
- **Squash merges allowed** for feature branches, regular merge for `develop` → `main`

---

## API Design Pattern

All endpoints follow this structure:

```python
@router.post("/endpoint")
async def endpoint(payload: InputSchema) -> OutputSchema:
    """
    Brief description of what this does.
    
    Request:
      payload (InputSchema): What goes in
    
    Response:
      OutputSchema: { "result": "...", "status": "success" }
    
    Raises:
      HTTPException: 400 if validation fails
      HTTPException: 500 if Gemini/Qdrant fails
    """
    try:
        result = await service.process(payload)
        return OutputSchema(result=result, status="success")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Testing Strategy (TDD)

1. Write test first (what should this do?)
2. Write minimal code to pass
3. Refactor for clarity + performance
4. Commit with `test:` prefix

Example:
```python
# test_pdf_extractor.py
def test_extract_valid_resume():
    pdf_path = "test_data/resume.pdf"
    profile = extract_resume(pdf_path)
    assert profile.skills is not None
    assert len(profile.skills) > 0
    assert profile.education_level in ["High School", "Bachelor's", "Master's", "PhD"]

def test_extract_invalid_pdf():
    with pytest.raises(ValueError):
        extract_resume("nonexistent.pdf")
```

---

## Folder Organization

```
backend/
├── app/
│   ├── __init__.py             # Empty
│   ├── main.py                 # FastAPI app + startup
│   ├── config.py               # Settings from env
│   ├── services/               # Business logic (no HTTP)
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py    # Pure functions
│   │   ├── llm_service.py
│   │   ├── vector_search.py
│   │   └── recommendation.py
│   ├── models/                 # Pydantic schemas only
│   │   ├── __init__.py
│   │   ├── user_profile.py
│   │   └── recommendation.py
│   └── routes/                 # HTTP handlers (thin layer)
│       ├── __init__.py
│       ├── health.py
│       ├── recommend.py
│       └── upload.py
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── unit/
│   │   ├── test_pdf_extractor.py
│   │   ├── test_llm_service.py
│   │   └── test_vector_search.py
│   └── integration/
│       ├── test_upload_endpoint.py
│       └── test_questionnaire_endpoint.py
├── requirements.txt
└── Dockerfile
```

**Why:** Separation of concerns. Services are testable functions; routes are thin handlers. You can test logic without HTTP.

---

## Debugging Approach

1. **Use logging, not print.** Set up `logging` module early.
2. **Structured logs.** Include context: `logger.info(f"Extracted {len(skills)} skills from {source}")`
3. **Error clarity.** Don't swallow exceptions; log stack trace + user-facing message.
4. **Local development.** Run with `--reload` + breakpoints in VSCode/Cursor.

---

## PR Review Checklist

Before pushing, ask yourself:
- [ ] Does the code do what the issue asks?
- [ ] Are there tests? Do they pass?
- [ ] Is it readable? Would a junior engineer understand it?
- [ ] Any obvious bugs or edge cases missed?
- [ ] Does it break anything else (run full test suite)?
- [ ] Is documentation updated (README, docstrings, PRD)?

---

## Learning Priorities (in order)

1. **FastAPI structure** — Routes, services, models separation
2. **Async/await patterns** — Make code non-blocking
3. **Testing with pytest** — Write tests first
4. **Git workflow** — Branches, commits, PRs, squash
5. **Error handling** — Graceful failures, logging
6. **Docker basics** — Package the app
7. **LLM integration** — Prompts, token usage, latency

---

## LLM Integration

- **Gemini 2.0 Flash**: Handles both extraction (resume → profile JSON) and ranking (top-N courses → explanations)
- **Why Gemini only**: Free tier sufficient for personal project; no API costs
- **You**: Decide prompts, handle failures, iterate on quality

When calling LLMs:
- Always handle rate limits (use retry logic with exponential backoff)
- Log requests/responses for debugging
- Never hard-code API keys (use env vars, .env file)
- Test with real data early (not just mock)

---

## Deployment Strategy (v1)

- **Local**: `uvicorn app.main:app --reload`
- **Docker**: Build image, run container
- **Cloud**: Deploy container to simple host (Render, Railway, Heroku)
- **Monitoring**: Add basic logging, error tracking (optional for MVP)

Full setup in `docs/SETUP.md` (to be written in Phase 0).

---

## Common Mistakes to Avoid

1. **Mixing business logic into routes.** → Use services/
2. **Hardcoding API keys.** → Use env vars via config.py
3. **Skipping error handling.** → Every external call can fail
4. **Tests that only pass "happy path".** → Test edge cases too
5. **PRs with no context.** → Always link to issue, explain "why"
6. **Committing debugging code.** → Use a linter (flake8) to catch cruft

---

## Quick Commands

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest                              # Run all tests
pytest tests/unit/ -v               # Run with verbose output
pytest --cov=app                    # Coverage report
uvicorn app.main:app --reload       # Dev server

# Git
git checkout -b feat/your-feature
git add .
git commit -m "feat: description of change"
git push origin feat/your-feature
# Open PR on GitHub

# Code quality
flake8 app/                         # Lint
black app/                          # Format
mypy app/                           # Type check
```

---

## Questions for Claude Code

- **"How would you structure X?"** → Get architecture advice, then you implement
- **"I'm stuck on Y, hints?"** → Get debugging direction, you fix
- **"Does this PR look good?"** → Get code review, address feedback, push again
- **"How do I test Z?"** → Get testing patterns, you write tests

**Not:** "Generate code for X" (ownership issue) — instead, "I'm building X, what's the pattern?"

---

## Next Session

This doc should be updated after each session with:
- New conventions discovered
- Gotchas hit and solved
- New learning (to remember for next project)

---

**Last Updated:** 2026-05-25
**Author:** Gautam (with Claude)
