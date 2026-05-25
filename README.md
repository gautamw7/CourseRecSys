# Course Recommendation System

A web application that recommends courses aligned to user skills and goals. Users provide a resume or answer a questionnaire, and receive personalized course recommendations with AI-generated explanations.

## Vision

Enable individuals of any age to discover and filter courses that match their skill level and learning goals, with intelligent recommendations powered by semantic search and LLM reasoning.

## Features (MVP)

- **Resume-based recommendations**: Upload a PDF resume → extract profile → get course matches
- **Fallback questionnaire**: 4-question form for users without a resume
- **AI-powered explanations**: Each recommendation includes "Why this course fits you"
- **Multi-source catalog**: Aggregates Coursera, NPTEL, and other course datasets
- **Semantic search**: Vector-based matching across course embeddings

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | React 18 + TypeScript | Type-safe, component-based UI; good DX for learning |
| **Backend** | FastAPI + Python 3.11 | Fast, async, well-documented; familiar to you |
| **Vector DB** | Qdrant (cloud or local) | Semantic search; proven in your prototypes |
| **LLM** | Gemini 2.0 Flash | Free tier; handles extraction + ranking |
| **Embedding Model** | SentenceTransformer (all-MiniLM-L6-v2) | Fast, CPU-friendly, good for education domain |
| **Parsing** | pdfplumber | Reliable PDF text extraction |
| **Deployment** | Docker + simple cloud host | Reproducible, portable |

## Project Structure

```
course-rec/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app entry
│   │   ├── config.py               # Settings (API keys, DB URLs)
│   │   ├── services/               # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── pdf_extractor.py    # Resume parsing
│   │   │   ├── llm_service.py      # Gemini 
│   │   │   ├── vector_search.py    # Qdrant queries
│   │   │   └── recommendation.py   # Ranking + explanation
│   │   ├── models/                 # Pydantic schemas
│   │   │   ├── __init__.py
│   │   │   ├── user_profile.py     # Resume Profile schema
│   │   │   └── recommendation.py   # Recommendation output schema
│   │   └── routes/                 # API endpoints
│   │       ├── __init__.py
│   │       ├── health.py           # GET /health
│   │       ├── recommend.py        # POST /recommend
│   │       └── upload.py           # POST /upload (resume)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ResumeUpload.tsx
│   │   │   ├── Questionnaire.tsx
│   │   │   ├── RecommendationCard.tsx
│   │   │   └── LoadingState.tsx
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   └── Results.tsx
│   │   ├── services/
│   │   │   └── api.ts              # API client
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
├── Data/
│   ├── Coursera Courses & Skills dataset 2024.csv
│   ├── Coursera Courses Dataset 2021.csv
│   ├── coursera_dataset_on_courses Random.csv
│   └── NPTEL Course List (July - Dec 2025).csv
├── docs/
│   ├── CONTEXT.md                  # Domain language + ubiquitous terms
│   ├── ARCHITECTURE.md             # System design + data flow
│   ├── PRD.md                      # Product requirements
│   ├── SETUP.md                    # Development setup
│   └── adr/                        # Architecture decision records
│       └── 001-vector-db-choice.md
├── .gitignore
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI/CD pipeline
├── CLAUDE.md                       # Project notes for Claude Code
└── .env.example
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (optional)
- API keys: Gemini, Claude (optional), Qdrant cloud

### Local Development

```bash
# 1. Clone repo
git clone <repo-url>
cd course-rec

# 2. Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
cp .env.example .env  # Update with your keys

# 3. Start backend
uvicorn app.main:app --reload

# 4. Frontend setup (in new terminal)
cd frontend
npm install
npm start

# App runs at http://localhost:3000
```

## Data Pipeline

1. **User Input** → Resume PDF or 4-question form
2. **Profile Extraction** → Gemini API extracts structured profile JSON
3. **Vector Encoding** → Profile converted to embedding (SentenceTransformer)
4. **Qdrant Search** → Find top-N matching courses (semantic similarity)
5. **Ranking + Explanation** → Claude generates "Why this matches" for top 5
6. **Response** → Frontend displays cards with course, score, explanation, link

## API Endpoints

```
POST /upload
  Body: { file: PDF }
  Response: { user_profile, recommendations }

POST /questionnaire
  Body: { goal, topic, level, time_per_week }
  Response: { user_profile, recommendations }

GET /health
  Response: { status: "ok" }
```

## Environment Variables

```
# Backend
GEMINI_API_KEY=...
QDRANT_URL=http://localhost:6333  # or cloud URL
QDRANT_API_KEY=...

# Frontend
REACT_APP_API_BASE_URL=http://localhost:8000
```

## Development Workflow

- **Branches**: `main` (production) → `develop` → feature branches (`feat/feature-name`)
- **Commits**: Conventional commits (`feat:`, `fix:`, `docs:`, etc.)
- **PRs**: Link to issue, describe changes, request review
- **Tests**: Run before pushing (unit + integration)

## Deployment

See `docs/SETUP.md` for production deployment (Docker, cloud hosting).

## Learning Goals

This project teaches:
- **Backend**: FastAPI, async patterns, API design, testing
- **Frontend**: React hooks, TypeScript, state management
- **DevOps**: Docker, CI/CD, git workflows
- **Project Management**: PRDs, issues, documentation, stakeholder communication
- **LLM Integration**: Working with Gemini API, prompt engineering
- **ML Ops**: Vector databases, embeddings, retrieval patterns

## Next Steps (v2+)

- [ ] Conversational questionnaire (multi-turn LLM chat)
- [ ] User accounts + saved recommendations
- [ ] Roadmap builder (goal-based learning paths)
- [ ] YouTube course integration
- [ ] Advanced filtering (price, duration, platform)

## Contributing

1. Pick an issue
2. Create a branch (`git checkout -b feat/your-feature`)
3. Implement + test
4. Open PR with clear description
5. Get reviewed, merge to `develop`

## License

MIT

## Contact

For questions or feedback, open an issue or reach out.