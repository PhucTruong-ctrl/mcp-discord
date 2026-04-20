# OpenClaw Discord Ecosystem Research Report
**Date:** March 14, 2026
**Status:** Public Ecosystem Analysis

---

## EXECUTIVE SUMMARY

The OpenClaw ecosystem has **5,366+ discoverable community skills** (from 13,700+ total in ClawHub after filtering spam/malware). Discord integrations are a first-class citizen with native channel support, dedicated skills marketplace, and production patterns from 300K+ GitHub stars.

**Key Finding:** Discord is positioned as both a *deployment target* (run agents as bots) AND an *orchestration surface* (use Discord as a coordination channel for multi-agent teams).

---

## I. CONCRETE CAPABILITY LIST

### A. Message & Interaction Operations

| Capability | Tool | Scope |
|-----------|------|-------|
| **Send messages** | discord.sendMessage | channels, threads, DMs; supports mentions, embeds, components |
| **Read messages** | discord.readMessages | bulk fetch (limit: 20/100); channel/thread scoped |
| **Edit/delete messages** | discord.editMessage, deleteMessage | same-author or bot-admin only |
| **React to messages** | discord.react | reactions + list users by emoji |
| **Manage threads** | discord.createThread, archiveThread | parent channel scoped; supports auto-archive |
| **Pin/unpin messages** | discord.pin, unpin | channel pinboard management |
| **Search messages** | discord.search | keyword-based; **limited by Discord native search** |

### B. Slash Commands & Components

| Capability | Pattern | Notes |
|-----------|---------|-------|
| **Slash commands** | native Discord API | OpenClaw bot auto-catalogs; contextual help integrated |
| **Mention patterns** | regex via config | `mentionPatterns` in group chat config; controls "is this a mention?" logic |
| **Quick replies** | pattern-match bypass | **PROPOSED:** zero-token auto-responses (pattern → static text) without LLM |
| **Interactive components** | buttons, select menus, modals | REST API via discord-hub skill; structured JSON payloads |

### C. Server Administration

| Capability | Tool | Scope |
|-----------|------|-------|
| **Channel management** | discord-admin CLI skill | create, update, delete with permission matrices |
| **Role management** | discord-admin CLI skill | role hierarchy, permission assignment, bulk operations |
| **Member actions** | moderation (gated) | invite, kick, ban, timeout; requires `moderation` action gate enabled |
| **AutoMod rules** | discord-admin CLI skill | configure and test server AutoMod policies |
| **Webhooks** | discord-admin CLI skill | provision, manage, test webhook flows |
| **Audit logs** | discord-admin CLI skill | inspect, filter, archive server history |
| **Events** | discord-admin CLI skill | scheduled event CRUD |

### D. Media & Custom Content

| Capability | Tool | Scope |
|-----------|------|-------|
| **Upload custom emoji** | emojiUpload action | PNG/JPG/GIF; ≤256KB; optional role-gating |
| **Upload stickers** | stickerUpload action | PNG/APNG/Lottie JSON; ≤512KB |
| **Send stickers** | sticker action | up to 3 per message; channel or DM |
| **Post polls** | poll action | 2–10 answers; 24-hr default duration; multi-select option |
| **Fetch permissions** | permissions action | bot permission check per channel |

### E. Information Retrieval

| Capability | Tool | Scope |
|-----------|------|-------|
| **Member info** | memberInfo action | username, roles, join date, presence |
| **Role info** | roleInfo action | role name, color, permissions |
| **Channel info** | channelInfo action | channel type, topic, permissions |
| **Voice status** | voiceStatus action | who's in voice; connected/connecting state |

### F. Semantic Search & Indexing

| Tool | Technology | Status |
|------|-----------|--------|
| **Discrawl** | Go CLI + SQLite FTS5 | **Production-ready** (v0.1.0 shipped); 502 stars |
| **Discord semantic search** | embedding-based vectors (OpenAI or local) | **RFC open** (#17875); proposed but not yet merged |

**Discrawl capabilities:**
- Mirror entire Discord guild into SQLite
- Maintain FTS5 search indexes for keyword queries
- Extract text from attachments into search index
- Record mentions as structured data for direct queries
- Live tail via Gateway events + periodic repair syncs
- Multi-guild ready; single-guild UX default
- Read-only SQL for ad-hoc analysis

---

## II. MODERATION WORKFLOWS

### Current Production Patterns (OpenClaw Server)

**Team Structure:**
```
Admin
├─ Discord Moderator Lead
│  └─ Discord Moderator Team (text channels)
├─ Voice Chat Lead
│  ├─ Voice Chat Team
│  └─ Event Presenters
├─ Helper Lead
│  └─ Helper Team (support channels)
└─ Configurator Lead
   └─ Configurator Team (bot/automod/permissions)
```

**Key Tools:**
- **discord-server-ctrl** skill: end-to-end server admin (channels, roles, AutoMod, audit logs, webhooks)
- **discord-profanity** skill: real-time content moderation (multi-language, configurable severity/thresholding)
- **Bot deployment:** Slack/Discord have native message content intents (Privileged Gateway Intents required)

**Action Gating** (security model):
```json
{
  "discord.actions.moderation": false,    // kick/ban/timeout — default disabled
  "discord.actions.roles": false,         // role add/remove — default disabled
  "discord.actions.stickers": true,       // all others enabled by default
  "discord.actions.polls": true
}
```

---

## III. FORUM & THREAD WORKFLOWS

### Session Isolation Pattern

**Problem:** Discord thread creation **forks session context** from parent channel → context pollution (#41823)

**Current Behavior:**
- Each Discord channel/thread gets isolated session
- Thread session inherits parent's conversation history
- **Root cause:** forked sessions carry unrelated context
- **Status:** Known issue; partial workarounds in config

**Best Practices (inferred):**
1. Use `discord.createThread` for topic isolation
2. Set `archive_duration` to auto-cleanup old threads
3. Monitor session memory with `/trim` or equivalent
4. For sensitive topics, create threads in dedicated support channels

**Available Controls:**
- `channels.discord.guilds[id].replyToMode` — control reply behavior (`"off"` vs default)
- `channels.discord.requireMention` — require explicit mentions in group contexts
- Message `message_reference` handling — thread replies can be toggled

---

## IV. RETRIEVAL & SEARCH PATTERNS

### A. Keyword Search (Native Discord)

**Limitations:**
- Discord search broken for "fuzzy" queries
- Keyword-only matching
- Poor thread discovery
- Recommendation: use Discrawl instead

### B. Discrawl Pattern (Recommended)

**Architecture:**
```
Discord Server
     ↓
[Discrawl Bot] ← reads via Bot Token
     ↓
  SQLite DB (local)
     ├─ messages table (FTS5 indexed)
     ├─ users table (structured)
     ├─ roles table (hierarchical)
     └─ mentions table (join index)
     ↓
[Ad-hoc SQL Queries] or [OpenClaw Semantic Search]
```

**Setup:**
```bash
discrawl config set-default-guild <guild-id>
discrawl sync                    # initial mirror
discrawl tail                    # live updates via Gateway
```

**Queries:**
```sql
-- Find messages about payment failures (month 3 of 2026)
SELECT * FROM messages 
WHERE content MATCH 'payment AND fail'
  AND timestamp > 1741747200;

-- Users mentioning specific role
SELECT user_id, message_id FROM message_mentions 
WHERE role_id = '<role-id>'
LIMIT 50;
```

**Cost:**
- Storage: ~1.5KB per message embedding (for semantic search variant)
- 100k messages ≈ 150MB storage
- Initial sync: depends on message count & intent limits
- Live tail: webhook + periodic repair (efficient)

### C. Semantic Search (RFC/Proposed)

**Vision (#17875):**
- Generate embeddings for each message (OpenAI or sentence-transformers)
- Store in SQLite with vector extension
- Query: `@agent find_in_discord("where did we discuss payment retry logic?")`
- Returns: top-k similar messages + links to threads

**Status:** Open RFC; not yet implemented in core but feasible via Discrawl + wrapper tool

---

## V. AGENT ORCHESTRATION PATTERNS

### A. Multi-Agent Coordination Mechanisms

| Mechanism | Use Case | Notes |
|-----------|----------|-------|
| **`sessions_send`** | Agent→Agent task handoff | A2A with requester context; target can reply |
| **`sessions_spawn`** | Spawn isolated sub-agents | Create temporary agent for single task |
| **`sessions_send` (reverse)** | Target replies to sender | Can cause duplicate messages (#39476 bug) |
| **Shared context files** | Team knowledge base | THESIS.md, SIGNALS.md, FEEDBACK-LOG.md |
| **Native topic routing** | Telegram/Discord topics → agentId | One bot, many internal agents |
| **Multi-bot routing** | Each agent = visible bot identity | Clear role separation; more overhead |

### B. Orchestration Patterns in Practice

#### Pattern 1: Lobster Workflow Engine

**What it does:**
- Deterministic multi-step automation (not LLM re-planning each step)
- YAML-based pipeline definition (version control friendly)
- Built-in approval gates before side effects
- Resumable state (halt → approve → resume)
- One call instead of many tool calls

**Example:**
```yaml
# email-triage.lobster
steps:
  - name: fetch_emails
    run: gmail.search query="has:flag"
  
  - name: draft_replies
    run: echo "Drafting replies..."
  
  - name: approval_gate
    wait: "approve?"
    
  - name: send_emails
    run: gmail.send
```

**Cost:** 1 LLM call (for draft) instead of 7 tool calls + planning overhead

#### Pattern 2: Orchestrator-Worker Hierarchy

**Setup (from raulvidis/openclaw-multi-agent-kit):**
```
Lead Agent (orchestrator)
├─ Coding Agent (primary for #dev-builds topic)
├─ QA Agent (secondary, triggered on handoff)
├─ DevOps Agent (tertiary, final deploy)
├─ Research Agent (specialized)
└─ Growth Agent (analytics/experiments)
```

**Flow:**
1. Lead Agent receives user request
2. Delegates to Coding Agent via `sessions_send`
3. Coding Agent builds, signals QA via handoff protocol
4. QA tests, if pass → signal DevOps
5. DevOps deploys; all post updates to shared channel

**Handoff Protocol:**
```
HANDOFF
from: coder
to: qa
task_id: build-142
priority: P1
summary: Validate checkout fix
context: branch=fix/coupon-rounding
deliver_to: discord:channel-id
deadline: asap
done_when:
- Repro no longer fails
- Regression checks pass
```

#### Pattern 3: Topic-Based Team Channels (Telegram/Discord)

**Discord Native Topic Routing:**
```json
{
  "channels": {
    "discord": {
      "guilds": {
        "server-id": {
          "topics": {
            "13": { "agentId": "connor" },  // internal agent
            "14": { "agentId": "kara" }
          }
        }
      }
    }
  }
}
```

**Behavior:**
- Topic 13 → routed to `connor` agent (internal)
- **But** message appears from the Discord bot's identity (shared bot for all topics)
- **Advantage:** one visible bot, many specialized brains
- **Caveat:** requires `agentId` to be allowed in config

### C. Webhook Integration for Event-Driven Orchestration

**Use case:** GitHub PR merged → trigger code review agent → post result to Discord

**Configuration:**
```json
{
  "hooks": {
    "enabled": true,
    "token": "shared-secret",
    "path": "/hooks",
    "allowedAgentIds": ["orchestrator", "main"]
  }
}
```

**Endpoints:**
- `POST /hooks/wake` — wake main agent, enqueue event
- `POST /hooks/agent` — route directly to specific agent with custom session key

**Example:**
```bash
curl -X POST http://localhost:18789/hooks/agent \
  -H 'Authorization: Bearer SECRET' \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Review: #4128 merged",
    "agentId": "code-reviewer",
    "sessionKey": "hook:github:pr-4128",
    "deliver": true,
    "channel": "discord"
  }'
```

---

## VI. NOTABLE DESIGN PATTERNS

### Pattern: Heartbeat Loop

**Concept:** Continuous autonomous operation backbone
- Periodic "ticks" that check for work
- Wake immediately if urgent event (webhook, mention)
- Handle idle gracefully (backoff, resource cleanup)
- Error recovery without manual intervention

**Discord Relevance:** Bot must stay responsive in channels while coordinating background tasks

### Pattern: Session Isolation

**Principle:** Each Discord channel/thread = separate conversational context
- Prevents context pollution (#41823 still open)
- Allows parallel conversations in same server
- Reduces token usage per conversation
- **Caveat:** thread forking from parent is a known issue

### Pattern: Action Gating

**Security model:** Fine-grained access control per agent
```json
{
  "discord.actions.moderation": false,     // disable risky actions by default
  "discord.actions.roles": false,
  "discord.actions.reactions": true        // safe actions enabled
}
```

### Pattern: Mention Pattern Matching

**Regex-based activation:** Control when bot responds in group contexts
```json
{
  "mentionPatterns": [
    "^(?!\\s*[!/\\?]).+"   // Match anything NOT starting with /, !, ?
  ]
}
```

**Caveat:** Patterns define what **counts as a mention** (trigger), not what to **ignore**

### Pattern: Shared Context via Markdown Files

**Team coordination without APIs:**
- `THESIS.md` → business direction (all agents read)
- `SIGNALS.md` → intelligence hub (research writes, all read)
- `FEEDBACK-LOG.md` → style corrections (live documentation)
- `SUPERGROUP-MAP.md` → topic↔agent mapping reference

---

## VII. SOURCE DOCUMENTATION & LINKS

### Official Documentation
- **OpenClaw Docs (Skills):** https://docs.openclaw.ai/tools/skills
- **Discord Channel Guide:** https://docs.openclaw.ai/channels/discord
- **Lobster Workflow Engine:** https://docs.openclaw.ai/tools/lobster
- **Webhooks:** https://docs.openclaw.ai/automation/webhook

### GitHub Repositories
- **openclaw/openclaw** (main) — 300K+ stars
  - Discord integration (#2453, #41823, #12943, #17875 — active issues)
  - RFC: Agent Teams (#10036)
- **openclaw/lobster** — Workflow engine (792 stars)
  - VISION.md: Safe automation with approval gates
- **openclaw/community** — Moderation policies + team structure (74 stars)
  - Mod Onboarding guide
  - Discord Moderator Lead responsibilities
- **VoltAgent/awesome-openclaw-skills** — Skill directory (36.8K stars)
  - 5,366 curated; 13,700+ total in ClawHub
- **raulvidis/openclaw-multi-agent-kit** — Multi-agent templates (222 stars)
  - Production 10-agent Telegram setup
  - Handoff protocols, soul templates, workspace organization
- **steipete/discrawl** — Discord→SQLite indexer (502 stars)
  - CLI + live tailing; v0.1.0 shipped
  - FTS5 search; multi-guild ready

### Skills Marketplace (LobeHub)
- **discord** (`steipete/discord`) — 8.1K stars
  - Messages, reactions, threads, pins, polls, moderation, member/role/channel info
- **discord-server-ctrl** (openclaw official) — 1.9K stars
  - Channel/role/member management, AutoMod, webhooks, audit logs
- **discord-admin** (openclaw official) — same as above
- **discord-hub** (openclaw official) — Bot API workflows via HTTPS
  - Interactions, commands, components; low-dependency
- **openclaw-profanity** (openclaw official) — Content moderation plugin
  - Multi-language detection; severity scoring; configurable actions

### Blog Posts & Guides
- **OpenClaw Blog:** https://openclawai.me (official)
- **Building Custom Skills:** https://openclawai.me/blog/building-skills
- **Beginner's Guide (Tencent Cloud):** https://www.tencentcloud.com/techpedia/139433
- **Multi-Agent Workflows (haimaker.ai):** https://haimaker.ai/blog/multi-agent-workflows-openclaw
- **Orchestration Patterns (Ken Huang):** https://kenhuangus.substack.com/p/openclaw-design-patterns-part-3-of
- **Deterministic Multi-Agent Pipelines (DEV):** https://dev.to/ggondim/how-i-built-a-deterministic-multi-agent-dev-pipeline-inside-openclaw-and-contributed-a-missing-4ool
- **How to Use Discrawl (Julian Goldie):** https://www.youtube.com/watch?v=KyXUJnYjjAo (YouTube)

### Real-World Examples
- **NextHello** (Michael Friedberg, Oracle) — Post-event conversation orchestration (DeepStation AI demo)
- **OpenClaw Multi-Agent Kit** (raulvidis) — 10-agent Telegram supergroup (production tested)
- **OpenClaw Community Server** — Moderation in practice (openclaw/community repo)

### Academic/Reference
- **Issue #17875** — Discord semantic search proposal (Feb 2026)
- **Discussion #10036** — RFC: Agent Teams coordination
- **Issue #39476** — A2A sessions_send duplicate handling
- **Issue #41823** — Thread session forking context pollution

### Ecosystem Tools
- **Composio** — OAuth + 1000+ app integrations (third-party)
- **You.com Search API** — Real-time web intelligence for agents
- **ClawHub CLI:** `clawhub install <skill-slug>`

---

## VIII. IMPLEMENTATION ROADMAP (For mcp-discord)

### Phase 1: Discoverable Skills Foundation
- [ ] Publish mcp-discord as OpenClaw skill to ClawHub
- [ ] Write SKILL.md with clear Discord use cases
- [ ] Create semantic-search-optimized descriptions

### Phase 2: Discord Operations Layer
- [ ] Integrate discord skill (steipete) as reference
- [ ] Implement missing capabilities (semantic search wrapper for Discrawl)
- [ ] Add profanity content moderation hooks

### Phase 3: Orchestration Ready
- [ ] Support Lobster workflow invocation from Discord
- [ ] Enable webhook-driven orchestration (GitHub→Discord→Multi-Agent)
- [ ] Add session isolation per channel/thread

### Phase 4: Multi-Agent Coordination
- [ ] Implement handoff protocol for Discord topics
- [ ] Add shared context file support (AGENTS.md, SIGNALS.md)
- [ ] Expose `sessions_send` for Discord-based coordination

### Phase 5: Production Hardening
- [ ] Add action gating (moderation safety)
- [ ] Implement mention pattern matching
- [ ] Add audit logging for sensitive actions

---

## IX. KNOWN GAPS & OPPORTUNITIES

| Gap | Priority | Owner | Status |
|-----|----------|-------|--------|
| Semantic search (embeddings) | Medium | community/contributor | RFC open (#17875) |
| Thread session isolation bug | High | OpenClaw core | Known (#41823) |
| A2A sessions_send duplicates | Medium | OpenClaw core | Proposed fix (#39476) |
| Zero-token quick replies | Low | backlog | RFC proposed (#44128) |
| Discord-native mention gating | High | docs/config | Documented but UX unclear |
| Multi-bot identity clarity | Medium | docs | Topic routing logic confusing |

---

**Report prepared:** 2026-03-14
**Researcher:** OpenClaw Ecosystem Scout
**Next Review:** Post mcp-discord Phase 1 launch
