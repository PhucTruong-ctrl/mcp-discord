## At a Glance
A quick feature overview to fix bugs reported from manual testing.

## Workstreams
- `Enum` `.value` bug caused by passing `None` down to `discord.py` internal methods. Fixed.
- `resolve_forum_post` missing in `DiscordGateway`. Fixed by aliasing to `resolve_thread`.
- `automod_get_ruleset` crashed because `ruleset.rules` missing from input failed `isinstance(list)` check. Fixed.

## Revision History
- 2026-04-28: Initial plan drafted and bugs fixed via shell. Tests pass.