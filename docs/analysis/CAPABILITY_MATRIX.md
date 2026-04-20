# OpenClaw Discord Capability Matrix

**Quick reference** for implementing Discord automation, moderation, search, and orchestration patterns.

---

## Message Operations

| Operation | Method | Availability | Cost | Gated? | Notes |
|-----------|--------|--------------|------|--------|-------|
| Send message | `discord.sendMessage` | ✅ GA | 1 call | ❌ No | Supports embeds, components, mentions |
| Edit message | `discord.editMessage` | ✅ GA | 1 call | ❌ No | Same-author or bot-admin only |
| Delete message | `discord.deleteMessage` | ✅ GA | 1 call | ❌ No | Requires permissions |
| Read messages | `discord.readMessages(limit:100)` | ✅ GA | 1 call | ❌ No | Channel/thread scoped; FTS via Discrawl |
| React (add emoji) | `discord.react` | ✅ GA | 1 call | ❌ No | Text emoji or custom |
| List reactions | `discord.reactions` | ✅ GA | 1 call | ❌ No | See users per emoji |
| Search messages | `discord.search` | ⚠️ Limited | 1 call | ❌ No | Keyword-only; use Discrawl for better UX |

---

## Thread & Organization

| Operation | Method | Availability | Cost | Gated? | Notes |
|-----------|--------|--------------|------|--------|-------|
| Create thread | `discord.createThread` | ✅ GA | 1 call | ❌ No | Parent channel scoped; supports auto-archive |
| Archive thread | `discord.archiveThread` | ✅ GA | 1 call | ❌ No | Hides from active list |
| Pin message | `discord.pin` | ✅ GA | 1 call | ❌ No | Channel pinboard |
| Unpin message | `discord.unpin` | ✅ GA | 1 call | ❌ No | Remove from pinboard |
| Post poll | `discord.poll` | ✅ GA | 1 call | ❌ No | 2–10 answers; 24-hr default; multi-select option |

---

## Server Administration

| Operation | Method | Availability | Cost | Gated? | Default | Notes |
|-----------|--------|--------------|------|--------|---------|-------|
| List channels | `discord.channelInfo` | ✅ GA | 1 call | ❌ No | — | Channel type, topic, permissions |
| Create channel | `discord-admin` skill | ✅ GA | script | ❌ No | — | With permission matrix |
| Update channel | `discord-admin` skill | ✅ GA | script | ❌ No | — | Name, topic, permissions |
| Delete channel | `discord-admin` skill | ✅ GA | script | ❌ No | — | Cascades threads |
| Create role | `discord-admin` skill | ✅ GA | script | ⛔ Yes | **Disabled** | Requires `discord.actions.roles: true` |
| Assign role | `discord-admin` skill | ✅ GA | script | ⛔ Yes | **Disabled** | Requires `discord.actions.roles: true` |
| Kick member | moderation action | ✅ GA | 1 call | ⛔ Yes | **Disabled** | Requires `discord.actions.moderation: true` |
| Ban member | moderation action | ✅ GA | 1 call | ⛔ Yes | **Disabled** | Requires `discord.actions.moderation: true` |
| Timeout member | moderation action | ✅ GA | 1 call | ⛔ Yes | **Disabled** | Requires `discord.actions.moderation: true` |
| View audit log | `discord-admin` skill | ✅ GA | script | ❌ No | — | Inspect, filter, archive |
| Configure AutoMod | `discord-admin` skill | ✅ GA | script | ❌ No | — | Test before enforcement |
| Manage webhooks | `discord-admin` skill | ✅ GA | script | ❌ No | — | Provision, manage, test |

---

## Member & Role Information

| Operation | Method | Availability | Notes |
|-----------|--------|--------------|-------|
| Get member info | `memberInfo` | ✅ GA | Username, roles, join date, presence |
| Get role info | `roleInfo` | ✅ GA | Name, color, permissions, hierarchy |
| Get channel permissions | `permissions` | ✅ GA | Bot permissions check per channel |
| Get voice status | `voiceStatus` | ✅ GA | Who's connected, connecting state |

---

## Custom Content

| Operation | Method | Max Size | Format | Availability | Notes |
|-----------|--------|----------|--------|--------------|-------|
| Upload emoji | `emojiUpload` | 256 KB | PNG/JPG/GIF | ✅ GA | Optional role-gating |
| Upload sticker | `stickerUpload` | 512 KB | PNG/APNG/Lottie JSON | ✅ GA | Requires description + tags |
| Send sticker | `sticker` | — | sticker ID | ✅ GA | Up to 3 per message |
| Send message with embeds | `sendMessage` | — | Discord embeds | ✅ GA | Rich formatting |
| Send button/select component | `discord-hub` skill (HTTPS) | — | JSON | ✅ GA | Via REST API (low-dependency) |

---

## Search & Indexing

### Native Discord Search
- **Method:** `discord.search`
- **Capabilities:** Keyword-only
- **Limitations:** Fuzzy matching broken; thread discovery poor
- **Recommendation:** Use Discrawl for better UX

### Discrawl (SQLite FTS5)
| Feature | Capability | Status |
|---------|-----------|--------|
| **Initial sync** | Mirror entire guild | ✅ Shipped (v0.1.0) |
| **Live updates** | Tail via Gateway events | ✅ Shipped |
| **Full-text search** | FTS5 indexes (keyword) | ✅ Shipped |
| **Mention indexing** | Structured user/role mentions | ✅ Shipped |
| **Semantic search** | Embedding-based vectors | ⏳ RFC open (#17875) |
| **Attachment indexing** | Extract text from files | ✅ Shipped |
| **Ad-hoc SQL** | Custom queries | ✅ Shipped |

**Setup:**
```bash
discrawl config set-default-guild <guild-id>
discrawl sync                    # initial mirror
discrawl tail                    # live updates
```

**Example query:**
```sql
SELECT * FROM messages 
WHERE content MATCH 'payment AND fail'
  AND timestamp > 1741747200;
```

**Storage:** ~1.5KB per message (for future embeddings); 100k messages ≈ 150MB

---

## Slash Commands & Interactions

| Feature | Capability | Status | Notes |
|---------|-----------|--------|-------|
| **Native commands** | `/command` | ✅ GA | Auto-cataloged; contextual help |
| **Command registry** | Dynamically register | ✅ GA | Via Discord Bot API |
| **Mention patterns** | Regex-based activation | ✅ GA | `mentionPatterns` config (trigger, not ignore) |
| **Quick replies** | Pattern → static text | ⏳ Proposed | Zero-token bypass (RFC #44128) |
| **Components** | Buttons, modals, select | ✅ GA | Via `discord-hub` skill (HTTPS) |

---

## Webhooks & Event Routing

| Endpoint | Use Case | Configuration | Status |
|----------|----------|---------------|--------|
| `POST /hooks/wake` | Wake main agent + enqueue event | `hooks.enabled: true` | ✅ GA |
| `POST /hooks/agent` | Route to specific agent | `hooks.allowedAgentIds: [...]` | ✅ GA |
| GitHub webhook | PR merged → code review agent | Via GitHub UI + OpenClaw listener | ✅ GA |
| Scheduled webhook | Cron-triggered orchestration | n8n, Zapier, or internal cron | ✅ GA |

**Example:**
```bash
curl -X POST http://localhost:18789/hooks/agent \
  -H 'Authorization: Bearer SECRET' \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Review PR #128",
    "agentId": "code-reviewer",
    "sessionKey": "hook:github:pr-128",
    "deliver": true,
    "channel": "discord"
  }'
```

---

## Multi-Agent Coordination

| Mechanism | Use Case | Availability | Notes |
|-----------|----------|--------------|-------|
| **`sessions_send`** | Agent→Agent handoff | ✅ GA | A2A with context; can cause duplicates (#39476) |
| **`sessions_spawn`** | Spawn isolated sub-agent | ✅ GA | Temporary agent for single task |
| **Shared context files** | Team knowledge base | ✅ GA | THESIS.md, SIGNALS.md, FEEDBACK-LOG.md |
| **Topic routing** | Discord topic → agentId | ✅ GA | One bot, many internal agents |
| **Handoff protocol** | Structured escalation | ✅ GA | Documented in multi-agent-kit |

---

## Workflows & Automation

| Pattern | Engine | Status | Cost Reduction |
|---------|--------|--------|-----------------|
| **Simple triggers** | Mention pattern matching | ✅ GA | — |
| **Multi-step automation** | Lobster (YAML) | ✅ GA | 80–90% token savings (deterministic vs LLM re-planning) |
| **Approval gates** | Lobster checkpoints | ✅ GA | Safety + cost control |
| **Resumable workflows** | Lobster state tokens | ✅ GA | Can pause indefinitely, resume exactly where left off |

**Lobster example:**
```yaml
steps:
  - name: fetch_emails
    run: gmail.search query="has:flag"
  
  - name: draft_replies
    run: email-draft-tool
  
  - name: approval
    wait: "approve?"
  
  - name: send
    run: gmail.send
```

---

## Security & Access Control

### Action Gating (Default Configuration)

```json
{
  "discord.actions": {
    "moderation": false,      // ⛔ disabled (kick/ban/timeout)
    "roles": false,           // ⛔ disabled (role add/remove)
    "reactions": true,        // ✅ enabled
    "stickers": true,         // ✅ enabled
    "polls": true,            // ✅ enabled
    "messages": true,         // ✅ enabled
    "threads": true,          // ✅ enabled
    "pins": true,             // ✅ enabled
    "search": true,           // ✅ enabled
    "memberInfo": true,       // ✅ enabled
    "roleInfo": true,         // ✅ enabled
    "channelInfo": true,      // ✅ enabled
    "voiceStatus": true,      // ✅ enabled
    "emojiUploads": true,     // ✅ enabled
    "stickerUploads": true    // ✅ enabled
  }
}
```

### Mention Pattern Matching

**Purpose:** Control when bot responds in group chats

```json
{
  "mentionPatterns": [
    "^(?!\\s*[!/\\?]).+"   // Match anything NOT starting with /, !, ?
  ]
}
```

**Important:** Patterns define what **counts as a mention** (triggers bot), not what to **ignore**.

---

## Session Isolation

| Context | Isolation | Behavior | Caveat |
|---------|-----------|----------|--------|
| **Channel** | ✅ Yes | Separate context per channel | — |
| **Thread** | ✅ Yes | Separate context per thread | ⚠️ Inherits parent history (#41823) |
| **DM** | ✅ Yes | Separate context per user | — |

**Best practice:** Use `/trim` to manage session memory; create support threads in dedicated channels to minimize inherited context.

---

## Profanity & Content Moderation

| Feature | Tool | Availability | Severity | Actions |
|---------|------|--------------|----------|---------|
| **Detection** | `openclaw-profanity` | ✅ GA | Configurable scoring | — |
| **Languages** | Multi-language support | ✅ GA | — | — |
| **Filtering** | Custom whitelist/blacklist | ✅ GA | — | Block, redact, warn, flag |
| **Enforcement** | Configurable per threshold | ✅ GA | — | Custom callbacks |

---

## Approval & Safety

| Feature | Status | Typical Use |
|---------|--------|-------------|
| **Approval gates** | ✅ Built into Lobster | Before sending, posting, or destructive ops |
| **Dry-run preview** | ✅ discord-admin skill | Test admin actions before execution |
| **Audit logging** | ✅ discord-admin skill | Inspect, filter, archive all moderation |
| **Action gating** | ✅ Config-based | Disable risky actions by default |

---

## Summary: What's Ready vs. What's Coming

| Category | Ready | Coming | Status |
|----------|-------|--------|--------|
| **Messages** | ✅ | — | Complete |
| **Threads** | ✅ | Isolation fix | Known issue #41823 |
| **Moderation** | ✅ | — | Available; gated by default |
| **Search** | ⚠️ Partial | Semantic | Discrawl v0.1 shipped; embedding RFC open |
| **Multi-agent** | ✅ | Duplication fix | A2A sessions_send duplicate bug #39476 |
| **Webhooks** | ✅ | — | Full event routing |
| **Workflows** | ✅ | — | Lobster production ready |

---

**Last Updated:** 2026-03-14
**Source:** [OPENCLAW_ECOSYSTEM_RESEARCH.md](../product/research/openclaw-ecosystem-research.md)
