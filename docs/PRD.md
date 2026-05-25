# Product Requirements Document (PRD)

**Course Recommendation System v1.0**

---

## 1. Executive Summary

**Problem:** Learners waste time searching for courses that match their skill level and goals. Existing platforms (Coursera, Udemy) have good search but no *personalized* recommendations. This system solves it by matching user profiles to courses via semantic search + AI reasoning.

**Solution:** A lightweight web app where users upload a resume or answer 4 questions, then receive 5 personalized course recommendations with explanations.

**Target Users:** Students, career-switchers, and upskilling professionals aged 18-50+.

**Scope:** MVP (stateless, no user accounts). v2+ adds conversational onboarding and saved recommendations.

---

## 2. Goals & Success Metrics

### Primary Goal
Enable users to discover relevant courses in <2 minutes from initial entry.

### Success Metrics (v1)
| Metric | Target | How Measured |
|--------|--------|--------------|
| **Time to Recommendations** | <2 min | Frontend analytics |
| **Recommendation Accuracy** | 70%+ user satisfaction | Post-request survey (v1.1) |
| **Cold-Start Clarity** | 95%+ understand input choice | UX testing |
| **System Uptime** | 99.5% | Monitoring (post-deploy) |

### Secondary Goals
- Teach project structure & practices (learning project for user)
- Explore Claude + Gemini API integration
- Build confidence in backend ownership

---

## 3. User Stories

### **Story 1: Resume-Based Recommendation**
```
As a career-switcher,
I want to upload my resume,
So that I get course recommendations aligned to my skills and goals.

Acceptance Criteria:
- PDF upload works (1-5MB files)
- Profile extraction completes in <10s
- Recommendations display within 5s of extraction
- Each recommendation shows title, platform, difficulty, score, link, explanation
```

### **Story 2: Questionnaire Fallback**
```
As a student without a strong resume,
I want to answer 4 questions about my goals and background,
So that I still get relevant recommendations.

Acceptance Criteria:
- 4-question form is clear and takes <1min to complete
- Question order makes sense (goal → topic → level → time)
- Answers map to structured profile JSON
- Recommendations display within 5s of submission
```

### **Story 3: Clear Recommendations**
```
As any user,
I want each recommendation to explain why it matches me,
So that I understand the recommendation rationale.

Acceptance Criteria:
- Each course card shows 1-sentence explanation
- Explanation references user's skills or goals
- User finds explanations helpful (subjective, but test in v1.1)
```

---

## 4. Features (MVP)

### **Core Features**
1. **Resume Upload** → Extract profile → Get recommendations
2. **Questionnaire Form** → 4 questions → Get recommendations
3. **Recommendation Display** → Cards with course info + explanation + link
4. **Health Check** → Backend availability monitoring

### **Out of Scope (v1)**
- [ ] User accounts / login
- [ ] Saved recommendations
- [ ] Conversational questioning (multi-turn chat)
- [ ] YouTube course integration
- [ ] Advanced filtering (price, duration, language)
- [ ] Admin dashboard

---

## 5. User Flow

```
Landing Page
    ↓
Choose Input Method
├─→ Upload Resume PDF
│   ├─→ Parse text (pdfplumber)
│   ├─→ Extract profile (Gemini)
│   ├─→ Encode to vector
│   ├─→ Search Qdrant
│   ├─→ Rank + explain (Claude)
│   └─→ Display Results
│
└─→ Answer Questionnaire
    ├─→ Goal? (dropdown)
    ├─→ Topic? (text input)
    ├─→ Level? (radio buttons)
    ├─→ Time/week? (slider)
    ├─→ Encode profile
    ├─→ Search Qdrant
    ├─→ Rank + explain (Claude)
    └─→ Display Results

Results Page
    ├─→ Show top-5 courses
    │   ├─→ Title
    │   ├─→ Platform (Coursera, NPTEL, etc)
    │   ├─→ Difficulty
    │   ├─→ Similarity %
    │   ├─→ Explanation
    │   └─→ Link to course
    │
    └─→ "Try again" button → back to Landing
```

---

## 6. Technical Requirements

### **Backend API**
| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/health` | GET | — | `{ status: "ok" }` |
| `/upload` | POST | PDF file | User profile + Recommendations |
| `/questionnaire` | POST | JSON (goal, topic, level, time) | User profile + Recommendations |

### **Performance**
- Profile extraction: <10s (Gemini API)
- Vector search: <500ms (Qdrant)
- Explanation generation: <5s (Claude API)
- Total time user-to-results: <20s (target <10s)

### **Data**
- Course datasets: 7000+ courses (Coursera + NPTEL)
- Vector embeddings: Precomputed, stored in Qdrant
- User profiles: Stateless (no DB required)

### **Security**
- API keys stored in `.env` (never in code)
- CORS enabled for frontend origin
- File upload restricted to PDFs, max 5MB
- No personal data persisted (v1)

---

## 7. Data Model

### **Resume Profile** (output of extraction)
```json
{
  "skills": ["Python", "SQL"],
  "domains_of_interest": ["ML", "Data Science"],
  "education_level": "Bachelor's",
  "experience_years": 2,
  "learning_goal": "career-switch",
  "time_per_week_hours": 5,
  "source": "resume | questionnaire"
}
```

### **Recommendation** (output of ranking)
```json
{
  "course_id": "coursera_123",
  "title": "ML with Python",
  "platform": "coursera",
  "difficulty": "intermediate",
  "provider": "Stanford",
  "similarity_score": 0.89,
  "explanation": "Your Python background and ML interest make this a strong fit.",
  "url": "https://coursera.org/..."
}
```

---

## 8. Success Criteria (Definition of Done)

A feature is "done" when:
- [ ] Code written, tested (unit + integration)
- [ ] PR reviewed and merged to `develop`
- [ ] Documentation updated (README, CONTEXT, code comments)
- [ ] No breaking changes to existing APIs
- [ ] Performance targets met (see Technical Requirements)
- [ ] Security checklist passed

---

## 9. Timeline (Phases)

### **Phase 0: Setup** (Week 1)
- [ ] GitHub repo + CI/CD
- [ ] Project structure (backend + frontend skeletons)
- [ ] Documentation (this PRD, CONTEXT.md, ARCHITECTURE.md)

### **Phase 1: Backend Core** (Week 2-3)
- [ ] PDF extraction + Gemini integration
- [ ] Qdrant client + vector search
- [ ] Claude ranking + explanation
- [ ] API endpoints `/upload`, `/questionnaire`
- [ ] Testing (unit + integration)

### **Phase 2: Frontend** (Week 3-4)
- [ ] Landing page (choose input method)
- [ ] Resume upload form
- [ ] Questionnaire form
- [ ] Results display (recommendation cards)
- [ ] Error handling + loading states

### **Phase 3: Polish & Deploy** (Week 4-5)
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Docker setup
- [ ] Deploy to cloud host
- [ ] Monitoring + error logging

### **Phase 4: v1.1 (Post-MVP)**
- [ ] User survey for accuracy feedback
- [ ] Minor UX tweaks
- [ ] Bug fixes

---

## 10. Open Questions / Risks

### **Q: What if Gemini API rate-limits?**
**A:** Batch requests; fallback to questionnaire. Add queuing in v1.1.

### **Q: How to handle PDF parsing failures?**
**A:** Graceful error + fallback to questionnaire form.

### **Q: Is 4-question questionnaire enough?**
**A:** Yes for MVP. v2 adds conversational follow-ups.

### **Q: What if Qdrant goes down?**
**A:** Return error to user; suggest trying again later.

---

## 11. Stakeholder Feedback / Approvals

- **Product**: [Your role]
- **Design**: [TBD — minimal custom design for MVP]
- **Backend**: [Your role — you own implementation]
- **Frontend**: [You + Claude co-development]
- **DevOps**: [Simple setup; can add monitoring later]

---

## 12. Appendix: Glossary

See `docs/CONTEXT.md` for all domain terms (Resume Profile, Recommendation, Course Document, etc.).
