# Claude Code Kickoff Prompt

Paste the message below as your **first message** in a fresh Claude Code session, after you have placed `CLAUDE.md`, `README.md`, and the `docs/` tree at the root of an empty directory.

---

Hi Claude. Read `CLAUDE.md` and the entire `docs/` directory before doing anything else, including all 9 ADRs in `docs/decisions/` and the HEC-RAS primer in `docs/hec-ras-primer.md`. Acknowledge what you've understood, flag anything ambiguous or that you would want to revisit, and then propose a concrete first action for **Week 1, Task 1: repository scaffold** (per `docs/roadmap.md` Phase A.0).

A few important things to know about how I want to work:

1. I have strong HEC-RAS / hydraulic engineering experience but I am new to ML and PyTorch. Teach as you go. When a new concept appears (tensor, autograd, U-Net, IoU, Dice loss, DataLoader, Lightning Module, etc.), give me a short conceptual primer *before* using it. Assume strong general engineering ability — just no ML background.

2. Walk me through each step rather than dumping a finished setup. I want to understand what we are building and why, not just end up with a working repo I cannot reason about.

3. Surface significant decisions before making them. If a choice has downstream consequences (architecture, dependency selection, sprint scope, repository conventions), present alternatives and tradeoffs, recommend, and let me decide. Then write the result as an ADR in `docs/decisions/` using `000-template.md` as the template.

4. The roadmap is a living document. If we learn something that changes our plan, update `docs/roadmap.md` and (if the change is significant) add an ADR.

5. Commit early and often, with conventional-commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

Start by confirming you've read the context and proposing the first action.
