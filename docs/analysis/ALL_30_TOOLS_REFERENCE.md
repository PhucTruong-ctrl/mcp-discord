# Complete Reference: All 30 Candidate Tools

**Generated**: March 14, 2026  
**Format**: Comprehensive lookup table with all tool metadata  

---

## Master Tool List (All 30 Candidates)

| # | Tool Name | Tier | Impact | Effort | Permission | Risk | Domain | Phase |
|---|-----------|------|--------|--------|-----------|------|--------|-------|
| **1** | get_audit_logs | T1 | ⭐⭐⭐⭐⭐ | 3h | VIEW_AUDIT_LOG | LOW | Audit | 1 |
| **2** | timeout_user | T1 | ⭐⭐⭐⭐⭐ | 2h | MODERATE_MEMBERS | LOW | Moderation | 1 |
| **3** | ban_user | T1 | ⭐⭐⭐⭐⭐ | 2h | BAN_MEMBERS | LOW | Moderation | 1 |
| **4** | kick_user | T1 | ⭐⭐⭐⭐⭐ | 2h | KICK_MEMBERS | LOW | Moderation | 1 |
| **5** | get_member_info | T1 | ⭐⭐⭐⭐⭐ | 2h | - | LOW | Members | 1 |
| **6** | create_role | T1 | ⭐⭐⭐⭐ | 2h | MANAGE_ROLES | LOW | Roles | 1 |
| **7** | delete_role | T1 | ⭐⭐⭐⭐ | 1h | MANAGE_ROLES | LOW | Roles | 1 |
| **8** | bulk_assign_roles | T1 | ⭐⭐⭐⭐ | 3h | MANAGE_ROLES | MEDIUM | Roles | 2 |
| **9** | list_webhooks | T1 | ⭐⭐⭐⭐ | 1h | MANAGE_WEBHOOKS | LOW | Webhooks | 2 |
| **10** | create_webhook | T1 | ⭐⭐⭐⭐ | 2h | MANAGE_WEBHOOKS | LOW | Webhooks | 2 |
| **11** | unban_user | T2 | ⭐⭐⭐⭐ | 1h | BAN_MEMBERS | LOW | Moderation | 1 |
| **12** | get_bans_list | T2 | ⭐⭐⭐⭐ | 1h | BAN_MEMBERS | LOW | Moderation | 2 |
| **13** | move_member_to_voice | T2 | ⭐⭐⭐⭐ | 2h | MOVE_MEMBERS | MEDIUM | Voice | 3 |
| **14** | set_channel_topic | T2 | ⭐⭐⭐⭐ | 1h | MANAGE_CHANNELS | LOW | Channels | 2 |
| **15** | get_channel_permissions | T2 | ⭐⭐⭐⭐ | 2h | MANAGE_CHANNELS | MEDIUM | Channels | 2 |
| **16** | set_channel_permissions | T2 | ⭐⭐⭐ | 3h | MANAGE_CHANNELS | MEDIUM-HIGH | Channels | 3 |
| **17** | pin_message | T2 | ⭐⭐⭐⭐ | 2h | PIN_MESSAGES | LOW | Messages | 1 |
| **18** | unpin_message | T2 | ⭐⭐⭐⭐ | 1h | PIN_MESSAGES | LOW | Messages | 2 |
| **19** | get_pinned_messages | T2 | ⭐⭐⭐⭐ | 1h | - | LOW | Messages | 2 |
| **20** | create_scheduled_event | T2 | ⭐⭐⭐ | 2h | MANAGE_GUILD | LOW | Events | 2 |
| **21** | get_scheduled_events | T3 | ⭐⭐⭐ | 1h | - | LOW | Events | 2 |
| **22** | delete_scheduled_event | T3 | ⭐⭐⭐ | 1h | MANAGE_GUILD | LOW | Events | 2 |
| **23** | execute_webhook | T3 | ⭐⭐⭐ | 2h | - | MEDIUM | Webhooks | 2 |
| **24** | get_member_roles | T3 | ⭐⭐⭐⭐ | 1h | - | LOW | Roles | 1 |
| **25** | mute_unmute_user | T3 | ⭐⭐⭐ | 2h | MUTE_MEMBERS | MEDIUM | Voice | 3 |
| **26** | disconnect_from_voice | T3 | ⭐⭐⭐ | 1h | MOVE_MEMBERS | LOW | Voice | 3 |
| **27** | set_automod_rules | T3 | ⭐⭐⭐ | 4h | MANAGE_GUILD | HIGH | Moderation | 3 |
| **28** | get_automod_status | T3 | ⭐⭐⭐ | 1h | MANAGE_GUILD | LOW | Moderation | 3 |
| **29** | get_invites | T3 | ⭐⭐ | 1h | MANAGE_GUILD | LOW | Guilds | 3 |
| **30** | create_invite | T3 | ⭐⭐ | 1h | CREATE_INSTANT_INVITE | LOW | Guilds | 3 |

---

## Tier Breakdown

### Tier 1: Highest Impact + Easiest Implementation (10 tools)
Best for Phase 1. Recommended: Start here.

| Tool | Use Case | Typical Query |
|------|----------|---------------|
| get_audit_logs | Show moderation history | "What actions were taken yesterday?" |
| timeout_user | Temporary suspension | "Timeout @user for 1 hour" |
| ban_user | Permanent ban | "Ban @user for spam" |
| kick_user | Remove from server | "Kick @user" |
| get_member_info | User details | "Show info on @user (join date, roles)" |
| create_role | New role | "Create 'Verified' role with blue color" |
| delete_role | Remove role | "Delete the 'test' role" |
| bulk_assign_roles | Mass assignment | "Add 'trusted' role to first 50 members" |
| list_webhooks | Webhook discovery | "List all webhooks in this server" |
| create_webhook | Webhook creation | "Create webhook for external service" |

### Tier 2: High Impact + Medium Complexity (10 tools)
Good for Phase 2. Builds on Phase 1.

| Tool | Use Case | Typical Query |
|------|----------|---------------|
| unban_user | Revoke ban | "Unban @user after review" |
| get_bans_list | Ban registry | "Show all banned users" |
| move_member_to_voice | Voice management | "Move @user to voice channel" |
| set_channel_topic | Channel description | "Set topic: 'General discussion'" |
| get_channel_permissions | Permission inspection | "Check what role can do in #general" |
| set_channel_permissions | Permission locks | "Lock #admin from @everyone" |
| pin_message | Message highlighting | "Pin this important message" |
| unpin_message | Remove pin | "Unpin that message" |
| get_pinned_messages | Pin list | "Show pinned messages in #rules" |
| create_scheduled_event | Event creation | "Schedule event for Saturday at 2pm" |

### Tier 3: Medium Impact + Moderate Complexity (10 tools)
Phase 3 and beyond. More specialized use cases.

| Tool | Use Case | Typical Query |
|------|----------|---------------|
| get_scheduled_events | Event list | "Show upcoming events" |
| delete_scheduled_event | Cancel event | "Cancel Saturday's event" |
| execute_webhook | External messages | "Send message via webhook" |
| get_member_roles | Role inspection | "List all roles for @user" |
| mute_unmute_user | Voice mute | "Mute @user in voice chat" |
| disconnect_from_voice | Voice kick | "Disconnect @user from voice" |
| set_automod_rules | Auto-moderation | "Block messages with slurs" |
| get_automod_status | Automod check | "Show current automod settings" |
| get_invites | Invite discovery | "List all active server invites" |
| create_invite | Invite creation | "Create 1-time invite link" |

---

## Permission Matrix

### By Permission Group

```
MODERATION GROUP (4 permissions, 6 tools):
  ├─ MODERATE_MEMBERS: timeout_user
  ├─ BAN_MEMBERS: ban_user, unban_user, get_bans_list
  └─ KICK_MEMBERS: kick_user

ROLE MANAGEMENT GROUP (1 permission, 4 tools):
  └─ MANAGE_ROLES: create_role, delete_role, bulk_assign_roles, get_member_roles

CHANNEL MANAGEMENT GROUP (1 permission, 3 tools):
  └─ MANAGE_CHANNELS: set_channel_topic, get_channel_permissions, set_channel_permissions

VOICE MANAGEMENT GROUP (2 permissions, 3 tools):
  ├─ MOVE_MEMBERS: move_member_to_voice, disconnect_from_voice
  └─ MUTE_MEMBERS: mute_unmute_user

MESSAGE MANAGEMENT GROUP (1 permission, 3 tools):
  └─ PIN_MESSAGES: pin_message, unpin_message

WEBHOOK GROUP (1 permission, 2 tools):
  └─ MANAGE_WEBHOOKS: create_webhook, list_webhooks

GUILD MANAGEMENT GROUP (1 permission, 4 tools):
  └─ MANAGE_GUILD: create_scheduled_event, delete_scheduled_event, set_automod_rules, get_automod_status

AUDIT GROUP (1 permission, 1 tool):
  └─ VIEW_AUDIT_LOG: get_audit_logs

INVITE GROUP (1 permission, 1 tool):
  └─ CREATE_INSTANT_INVITE: create_invite

READ-ONLY TOOLS (6 tools - no permissions required):
  ├─ get_member_info
  ├─ get_scheduled_events
  ├─ get_pinned_messages
  ├─ execute_webhook (external auth)
  ├─ get_invites
  └─ get_automod_status
```

---

## Implementation Dependencies

### Tools That Enable Other Tools

```
FOUNDATION TOOLS (must do first):
  get_member_info → unblock: all moderation/inspection tools
  get_member_roles → unblock: bulk_assign_roles, permission checks

MODERATION FOUNDATION:
  ban_user → unblock: unban_user, get_bans_list
  
ROLE FOUNDATION:
  create_role → unblock: bulk_assign_roles
  
WEBHOOK FOUNDATION:
  list_webhooks → unblock: execute_webhook

CHANNEL FOUNDATION:
  get_channel_permissions → unblock: set_channel_permissions
```

---

## Estimated Implementation Effort

| Phase | Tools | Duration | Effort per Tool | Total Effort |
|-------|-------|----------|-----------------|--------------|
| **Phase 1** | 10 | Week 1-3 | 1.8h avg | 18 hours |
| **Phase 2** | 10 | Week 4-6 | 2.0h avg | 20 hours |
| **Phase 3** | 10 | Week 7-10 | 3.5h avg | 35 hours |
| **TOTAL** | **30** | ~10 weeks | **2.4h avg** | **73 hours** |

**Testing overhead**: Add 50% for QA/integration testing

---

## Risk Profile by Tool

```
🟢 LOW RISK (18 tools):
   - Use well-established Discord APIs
   - No external dependencies
   - Built-in Discord rate limiting
   - Predictable permission system
   
   Tools: get_audit_logs, timeout_user, ban_user, kick_user, 
          unban_user, get_member_info, create_role, delete_role,
          get_member_roles, list_webhooks, pin_message, unpin_message,
          get_pinned_messages, set_channel_topic, get_scheduled_events,
          disconnect_from_voice, get_invites, create_invite

🟡 MEDIUM RISK (8 tools):
   - Batch operations or complex state changes
   - More configuration points
   - Rate limiting considerations
   - Tested in production but need validation
   
   Tools: bulk_assign_roles, get_bans_list, move_member_to_voice,
          get_channel_permissions, set_channel_permissions,
          create_scheduled_event, delete_scheduled_event, execute_webhook

🟠 MEDIUM-HIGH RISK (3 tools):
   - Complex API interactions
   - Permanent data modifications
   - Requires careful error handling
   - Needs extensive testing
   
   Tools: set_automod_rules, get_automod_status, mute_unmute_user

🔴 HIGH RISK (1 tool):
   - Highly configurable with many failure modes
   - Significant API complexity
   - Requires advanced testing
   
   Tools: set_automod_rules (if implementing advanced filtering)
```

---

## Success Criteria per Phase

### Phase 1 Success
- ✅ All 10 tools implemented and passing unit tests
- ✅ Integration tests for moderation workflow passing
- ✅ Live server testing on test Discord guild
- ✅ Permission validation working
- ✅ Rate limiting verified

### Phase 2 Success
- ✅ Webhook creation/execution working end-to-end
- ✅ Scheduled events CRUD complete
- ✅ Bulk operations functional
- ✅ Permission overwrites tested
- ✅ No regression in Phase 1 tools

### Phase 3 Success
- ✅ Voice channel operations stable
- ✅ Automod rules functional
- ✅ All 30 tools documented
- ✅ Comprehensive test coverage (>80%)
- ✅ Production-ready enterprise features

---

## Next Actions

1. **Immediate** (Today): Review TOOL_GAP_ANALYSIS.md and PHASE1_IMPLEMENTATION.md
2. **This Week**: Get sign-off on Phase 1 scope and timeline
3. **Week 1**: Set up test Discord server and schema validation
4. **Week 1-2**: Implement Moderation tier tools (1-6)
5. **Week 2-3**: Implement Roles/Audit tier tools (7-10)

---

**Total Analysis**: 730 lines of documentation + 30 tool specifications + dependency matrix + 3-phase roadmap
