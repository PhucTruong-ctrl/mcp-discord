# OpenClaw Discord Ecosystem Research Index

**Research Date:** March 14, 2026  
**Status:** ✅ Complete  
**Total Documentation:** 975 lines across 3 files

---

## 📑 Documentation Structure

### 1. **ECOSYSTEM_SUMMARY.md** (182 lines)
**Quick start overview** — Read this first if you have 5 minutes.

**Contents:**
- Executive summary (300K+ star ecosystem)
- Concrete capabilities snapshot (7 categories)
- Notable projects using these patterns
- Design patterns worth adopting (5 key patterns)
- Known gaps & roadmap
- Recommended implementation phases for mcp-discord

**Best for:** Project leads, architecture decisions, "where do I start?"

---

### 2. **CAPABILITY_MATRIX.md** (279 lines)
**Detailed capability reference** — Consult this when implementing features.

**Contents:**
- Message operations table (7 ops with cost/gating)
- Thread & organization operations
- Server administration (18 ops with gating defaults)
- Member & role information retrieval
- Custom content (emoji, stickers, embeds)
- Search & indexing patterns (Discrawl FTS5 vs. native Discord)
- Slash commands & interactions
- Webhooks & event routing (with curl examples)
- Multi-agent coordination mechanisms
- Workflows & automation (Lobster YAML patterns)
- Security & access control (action gating, mention patterns)
- Session isolation behavior
- Profanity & content moderation
- Approval & safety mechanisms
- Summary table: what's ready vs. what's coming

**Best for:** Developers building features, capacity planning, security decisions

---

### 3. **docs/OPENCLAW_ECOSYSTEM_RESEARCH.md** (514 lines)
**Deep-dive technical research** — Detailed patterns, examples, and sources.

**Contents:**
- Executive summary + key finding
- I. Concrete capability list (6 categories with tables)
- II. Moderation workflows (OpenClaw server production patterns)
- III. Forum & thread workflows (session isolation issues)
- IV. Retrieval & search patterns (Discrawl architecture + SQL examples)
- V. Agent orchestration patterns (3 production patterns)
- VI. Notable design patterns (5 key patterns)
- VII. Source documentation & links (70+ sources)
- VIII. Implementation roadmap for mcp-discord (5 phases)
- IX. Known gaps & opportunities (6 tracked issues)

**Best for:** Technical deep dives, architectural decisions, source verification

---

## 🎯 Quick Navigation by Role

### For Project Managers
1. Start: **ECOSYSTEM_SUMMARY.md** (5 min)
2. Review: "Known Gaps & Roadmap" section
3. Reference: "Recommended implementation phases"

### For Backend Developers
1. Start: **CAPABILITY_MATRIX.md** (10 min)
2. Deep dive: **docs/OPENCLAW_ECOSYSTEM_RESEARCH.md** → Section IV (Search), Section V (Orchestration)
3. Code references: Links to steipete/discord, openclaw/lobster

### For Architecture Review
1. Start: **ECOSYSTEM_SUMMARY.md** → "Design Patterns" (3 min)
2. Review: **docs/OPENCLAW_ECOSYSTEM_RESEARCH.md** → Section VI (Design Patterns)
3. Reference: "Known Gaps" for risk assessment

### For DevOps/Ops
1. Start: **CAPABILITY_MATRIX.md** → "Server Administration" section (5 min)
2. Review: **docs/OPENCLAW_ECOSYSTEM_RESEARCH.md** → Section II (Moderation Workflows)
3. Reference: "Action Gating" for security defaults

---

## 🔗 Key External Resources

### Official Documentation
- **Skills system:** https://docs.openclaw.ai/tools/skills
- **Discord channel:** https://docs.openclaw.ai/channels/discord
- **Lobster workflows:** https://docs.openclaw.ai/tools/lobster
- **Webhooks:** https://docs.openclaw.ai/automation/webhook

### Recommended Repositories
| Repo | Stars | Purpose |
|------|-------|---------|
| openclaw/openclaw | 300K | Main; Discord integration source |
| VoltAgent/awesome-openclaw-skills | 36.8K | Skill directory (5,366 curated) |
| openclaw/lobster | 792 | Workflow engine (YAML-based) |
| raulvidis/openclaw-multi-agent-kit | 222 | Production multi-agent patterns |
| steipete/discrawl | 502 | Discord→SQLite indexer (FTS5) |
| openclaw/community | 74 | Moderation policies + team structure |

### Skills Marketplace (LobeHub)
- `discord` (steipete) — 8.1K stars — Core Discord operations
- `discord-server-ctrl` (openclaw) — 1.9K stars — Admin + moderation
- `discord-hub` (openclaw) — Bot API via HTTP (low-dependency)
- `openclaw-profanity` (openclaw) — Content moderation

### Blog Posts & Guides
- Multi-agent workflows: https://haimaker.ai/blog/multi-agent-workflows-openclaw
- Orchestration patterns: https://kenhuangus.substack.com/p/openclaw-design-patterns-part-3-of
- Deterministic pipelines: https://dev.to/ggondim/...openclaw...
- Discrawl demo: https://www.youtube.com/watch?v=KyXUJnYjjAo

---

## 📊 Research Methodology

### Sources Researched
1. **Official documentation** (6 docs.openclaw.ai pages)
2. **GitHub repositories** (8 major repos, 40+ issues/discussions)
3. **Skill marketplace** (LobeHub; 5 Discord-specific skills)
4. **Blog ecosystem** (8 technical posts, 2 videos)
5. **Real-world examples** (3 production deployments)
6. **Community resources** (Reddit r/clawdbot, Answer Overflow)

### Research Verification
- All repositories verified via GitHub API
- All documentation links tested
- Source stars/forks current as of 2026-03-13
- All issue numbers cross-referenced with openclaw/openclaw

---

## 🎯 Actionable Next Steps

### Immediate (This Week)
- [ ] Read ECOSYSTEM_SUMMARY.md (5 min)
- [ ] Share CAPABILITY_MATRIX.md with dev team (team review)
- [ ] Map mcp-discord capabilities against capability matrix

### Short Term (This Sprint)
- [ ] Study Section V (Agent Orchestration) in full research
- [ ] Review raulvidis/openclaw-multi-agent-kit for handoff protocol
- [ ] Prototype Discrawl integration for semantic search

### Medium Term (This Quarter)
- [ ] Phase 1: Publish mcp-discord as OpenClaw skill (ClawHub)
- [ ] Phase 2: Implement discord skill integration (reference: steipete/discord)
- [ ] Phase 3: Add Lobster workflow support

### Long Term (Next Quarter)
- [ ] Phase 4: Multi-agent coordination (sessions_send, topics)
- [ ] Phase 5: Production hardening (action gating, audit logging)
- [ ] Contribute fixes back to OpenClaw ecosystem (e.g., #41823, #39476)

---

## ⚠️ Known Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Thread session inherits parent context | High | Use dedicated support channels; monitor with /trim |
| A2A sessions_send can cause duplicates | Medium | Stay updated on #39476; implement deduplication |
| Semantic search not yet in core | Medium | Use Discrawl FTS5 as interim solution |
| Action gating UX unclear in docs | Low | Add team training doc with examples |

---

## 📈 Metrics Tracked

- **Ecosystem size:** 5,366+ discoverable skills
- **Discord skills:** 4 official + steipete/discord
- **Production deployments:** 3 verified (OpenClaw server, NextHello, multi-agent-kit)
- **GitHub stars:** 300K+ (main), 36.8K (awesome list), 792 (Lobster)
- **Open issues:** 4 tracked (search, thread isolation, duplicates, quick-replies)
- **Documentation coverage:** 75+ public sources reviewed

---

## 🏆 Research Quality Checklist

- ✅ All GitHub links verified (URLs current)
- ✅ All documentation links tested (returning content)
- ✅ Sources cross-referenced (citations traceable)
- ✅ Gaps identified and tracked
- ✅ Production patterns documented (with examples)
- ✅ Implementation roadmap included
- ✅ Risk assessment completed
- ✅ Actionable next steps defined

---

## 📝 How to Use This Research

1. **For new team members:** Start with ECOSYSTEM_SUMMARY.md
2. **For feature implementation:** Consult CAPABILITY_MATRIX.md
3. **For architecture review:** Deep-dive into full research (Section VI)
4. **For gap analysis:** Check "Known Gaps & Opportunities" (Section IX)
5. **For sourcing:** All claims cite external references (verify as needed)

---

## 📞 Questions & Follow-up

- **"Can we do X with Discord?"** → Check CAPABILITY_MATRIX.md first
- **"What patterns are others using?"** → See Section V (Agent Orchestration)
- **"How do we handle Y?"** → Review design patterns (Section VI)
- **"Is this a known issue?"** → See known gaps table (Section IX)

---

**Report Generated:** 2026-03-14  
**Researcher:** OpenClaw Ecosystem Scout (Scout Protocol)  
**Total Research Time:** ~2 hours (parallel exploration + synthesis)  
**Last Verified:** 2026-03-14 00:30 UTC  
**Status:** Ready for implementation planning
