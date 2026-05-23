# ADR 008: Collaboration Mode — Claude Code in VS Code

**Status:** Accepted
**Date:** 2026-05-21

## Context

For a long-running multi-phase project like this, there are several ways the user can collaborate with Claude:

- Claude.ai web/mobile interface, with the user manually copying code into their workstation
- Claude Code in the terminal or VS Code, with direct file system and command execution access
- Some hybrid

The web interface is convenient for high-level planning but cannot execute code, manage files, or maintain durable repo state. Claude Code can do all of that, plus the same reasoning and planning.

## Decision

All work — planning, decisions, optioneering, note-taking, execution — happens in **Claude Code within VS Code**. This repository is the system of record. The web interface is used only for occasional mobile access if needed.

### Operational conventions

- **CLAUDE.md** at the repo root is the canonical project context, auto-loaded by Claude Code on every session.
- **`docs/decisions/`** captures significant decisions as ADRs (this file is an example).
- **`docs/roadmap.md`** is the living phased plan.
- **`notes/`** is a scratchpad for ad-hoc thinking, retrospectives, and "what surprised me this week" entries.
- All ML and engineering work happens in the repo with regular commits; nothing important lives only in conversation history.
- When starting a new Claude Code session, the user does not need to re-explain context; the repo (read by Claude Code on session start) carries it.

## Consequences

### Positive
- Single source of truth: everything lives in the repo.
- No context loss between sessions or between devices.
- Decisions, roadmap, and code are all version-controlled together.
- Future contributors (human or AI) can be onboarded by reading the repo.

### Negative / risks
- Loss of convenient mobile access for ad-hoc planning conversations (mitigated by using the web interface for that when needed, then writing the result back into the repo).
- Requires discipline to keep `CLAUDE.md`, ADRs, and the roadmap updated — stale context is worse than no context.

## References

- Claude Code docs: https://docs.claude.com/en/docs/claude-code/overview
- Claude Code setup: https://docs.claude.com/en/docs/claude-code/setup
