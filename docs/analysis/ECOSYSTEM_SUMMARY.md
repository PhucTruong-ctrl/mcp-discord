# OpenClaw Discord Ecosystem Summary

**Date:** March 14, 2026 | **Status:** Research Complete

## Quick Navigation

- 📋 **Full Research:** [OPENCLAW_ECOSYSTEM_RESEARCH.md](../product/research/openclaw-ecosystem-research.md) (514 lines)
- 🔗 **Skills Directory:** [VoltAgent/awesome-openclaw-skills](https://github.com/VoltAgent/awesome-openclaw-skills) (5,366+ skills)
- 📚 **Official Docs:** [docs.openclaw.ai](https://docs.openclaw.ai)

---

## Executive Summary

OpenClaw (300K+ GitHub stars) treats Discord as a **first-class deployment platform** with native bot channel support. The ecosystem includes:

- **5,366+ discoverable community skills** (13,700+ total in ClawHub)
- **Production Discord integrations** (OpenClaw server runs moderation at scale)
- **Deterministic workflow engine** (Lobster) for multi-step automations
- **Advanced search capabilities** (Discrawl CLI indexes servers into SQLite)
- **Multi-agent coordination patterns** proven in live deployments

---

## Concrete Capabilities

### Discord Operations (Ready to Use)

| Category | Capability | Status |
|----------|-----------|--------|
| **Messages** | Send/edit/delete/read/search | ✅ Production |
| **Reactions** | React, list users, emoji management | ✅ Production |
| **Threads** | Create, archive, manage context | ✅ Production (context isolation TBD) |
| **Moderation** | Member actions, roles, AutoMod, audit logs | ✅ Production (gated by default) |
| **Content** | Custom emoji, stickers, polls | ✅ Production |
| **Search** | Keyword (limited), semantic (proposed) | ⚠️ Partial |
| **Webhooks** | Inbound event routing to agents | ✅ Production |

### Advanced Patterns (Novel to OpenClaw)

| Pattern | Purpose | Example |
|---------|---------|---------|
| **Lobster Workflows** | Deterministic multi-step automation | Email triage → approval → send (1 LLM call) |
| **Multi-Agent Coordination** | Agent-to-agent handoff via Discord topics | Code → Review → QA → Deploy pipeline |
| **Session Isolation** | Per-channel/thread conversation context | Parallel conversations, reduced tokens |
| **Discrawl Indexing** | Full-text search via SQLite | Query 100k messages instantly (local) |
| **Action Gating** | Fine-grained security (moderation disabled by default) | `discord.actions.moderation: false` |
| **Webhooks + Topics** | Event-driven agent routing (GitHub → Discord → Agent) | PR merged → code-reviewer agent |

---

## Notable Projects Using These Patterns

1. **OpenClaw Community Server** (openclaw/community)
   - 4-team moderation structure (Discord, Voice, Helper, Configurator)
   - Hierarchical escalation model
   - Source: https://github.com/openclaw/community

2. **NextHello** (Michael Friedberg, Oracle)
   - Real-world agent swarm demo
   - Post-event conversation orchestration
   - Source: DeepStation AI demo

3. **OpenClaw Multi-Agent Kit** (raulvidis)
   - Production 10-agent Telegram setup (Discord adaptable)
   - Handoff protocol + soul templates
   - Source: https://github.com/raulvidis/openclaw-multi-agent-kit

4. **Discrawl** (steipete)
   - Discord→SQLite indexer with FTS5
   - v0.1.0 shipped; 502 stars
   - Source: https://github.com/steipete/discrawl

---

## Design Patterns Worth Adopting

### 1. **Heartbeat Loop**
Continuous autonomous operation with event-driven wake-ups (webhooks, mentions).

### 2. **Session Isolation Per Channel**
Each Discord channel/thread = separate conversational context → prevents context pollution.

### 3. **Action Gating (Security)**
```json
{
  "discord.actions.moderation": false,    // disabled by default
  "discord.actions.roles": false,
  "discord.actions.stickers": true        // safe actions enabled
}
```

### 4. **Shared Context via Markdown**
Team coordination without APIs:
- `THESIS.md` — business direction
- `SIGNALS.md` — intelligence hub
- `FEEDBACK-LOG.md` — live style corrections

### 5. **Orchestrator-Worker Hierarchy**
One lead agent spawns specialists; all post updates to shared channels.

---

## Known Gaps & Roadmap

| Issue | Priority | Status | Reference |
|-------|----------|--------|-----------|
| **Semantic search for Discord** | Medium | RFC open | #17875 |
| **Thread session isolation bug** | High | Known | #41823 |
| **A2A sessions_send duplicates** | Medium | Proposed fix | #39476 |
| **Zero-token quick replies** | Low | RFC | #44128 |

---

## For mcp-discord Implementation

### Recommended Phases

**Phase 1:** Publish as OpenClaw skill to ClawHub
- Write SKILL.md with semantic-search-optimized descriptions
- Add Discord use cases (moderation, multi-agent coordination, search)

**Phase 2:** Discord operations layer
- Integrate steipete/discord skill as reference
- Implement Discrawl wrapper for semantic search

**Phase 3:** Orchestration ready
- Support Lobster workflow invocation from Discord
- Webhook-driven GitHub→Discord→Multi-Agent automation

**Phase 4:** Multi-agent coordination
- Handoff protocol for Discord topics
- Shared context file support (AGENTS.md, SIGNALS.md)

**Phase 5:** Production hardening
- Action gating (moderation safety)
- Mention pattern matching
- Audit logging

---

## Key Resources

### Skill Marketplace (Production Skills)
- `discord` (steipete) — 8.1K stars — messages, reactions, threads, moderation
- `discord-server-ctrl` (openclaw) — 1.9K stars — server admin
- `discord-hub` (openclaw) — Bot API workflows via HTTP
- `openclaw-profanity` (openclaw) — Content moderation

### Official Repositories
- **openclaw/openclaw** — Main (300K stars)
- **openclaw/lobster** — Workflow engine (792 stars)
- **openclaw/community** — Moderation policies (74 stars)
- **VoltAgent/awesome-openclaw-skills** — Skill directory (36.8K stars)

### Documentation
- Skills: https://docs.openclaw.ai/tools/skills
- Discord: https://docs.openclaw.ai/channels/discord
- Lobster: https://docs.openclaw.ai/tools/lobster
- Webhooks: https://docs.openclaw.ai/automation/webhook

### Blog & Guides
- Multi-agent workflows: https://haimaker.ai/blog/multi-agent-workflows-openclaw
- Orchestration patterns: https://kenhuangus.substack.com/p/openclaw-design-patterns-part-3-of
- Deterministic pipelines: https://dev.to/ggondim/how-i-built-a-deterministic-multi-agent-dev-pipeline-inside-openclaw-and-contributed-a-missing-4ool
- Discrawl video: https://www.youtube.com/watch?v=KyXUJnYjjAo

---

## Next Steps

1. ✅ **Research complete** — Read [OPENCLAW_ECOSYSTEM_RESEARCH.md](../product/research/openclaw-ecosystem-research.md)
2. 📋 **Review Phase 1 implementation plan** — Skills marketplace publication
3. 🔗 **Map mcp-discord capabilities to OpenClaw patterns** — Identify gaps
4. 🛠️ **Prototype integration** — discord skill + Discrawl wrapper
5. 🚀 **Publish to ClawHub** — Make discoverable ecosystem-wide

---

**Generated by:** OpenClaw Ecosystem Scout (Scout Protocol)
**Last Updated:** 2026-03-14
**Repository:** https://github.com/PhucTruong-ctrl/mcp-discord
