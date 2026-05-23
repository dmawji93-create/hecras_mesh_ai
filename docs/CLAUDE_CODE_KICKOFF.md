# Claude Code Kickoff Prompt

Paste the message below as your **first message** in a Claude Code session to resume the project. The repository already exists and Phase A.0 Week 1 is complete — this is a resume prompt, not a from-scratch scaffold.

---

Hi Claude. This is an ongoing project. Before doing anything else, read, in this order:

1. `CLAUDE.md` — canonical project context.
2. `docs/STATUS.md` — current status.
3. `docs/hec-ras-primer.md` — domain background.
4. Every ADR in `docs/decisions/` — there are thirteen files, `000-template.md` through `012-refinement-numerical-first.md`. Note that ADR 003 carries a dated amendment.
5. `docs/roadmap.md` — the strategic phased plan.
6. Every file in `docs/build-plan/` in order, `00` through `09` — the executable, checkpointed plan.

Then acknowledge what you understand and flag anything ambiguous or inconsistent.

Project state: Phase A.0 Week 1 (plumbing) is complete. The next work is **build-plan Stage 1 — Feature & Label Pipeline**.

**First action:** update `docs/STATUS.md` so it reflects current reality — there are now 12 ADRs (001-012), the `docs/build-plan/` directory exists, and the next step is build-plan Stage 1 (which corresponds to the old roadmap "Week 2"). Then propose a concrete first action for Stage 1.

How I want to work:

1. I have strong HEC-RAS / hydraulic engineering experience but I am new to ML and PyTorch. Teach as you go — when a new concept appears (tensor, autograd, U-Net, IoU, Dice loss, DataLoader, LightningModule, etc.), give a short conceptual primer before using it. Assume strong general engineering ability, no ML background.

2. Walk me through each step rather than dumping a finished setup. I want to understand what we are building and why.

3. Surface significant decisions before making them — present alternatives and tradeoffs, recommend, let me decide. Then record the outcome as an ADR in `docs/decisions/` using `000-template.md`.

4. The roadmap and build plan are living documents. If we learn something that changes the plan, update them, and add an ADR if the change is significant.

5. Do not start a build-plan stage until the previous stage's checkpoint criteria are all met and verified.

6. Commit early and often, with conventional-commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

7. Be candid — flag risks and uncertainty honestly.

Start by confirming you've read the context, then proceed with the first action above.
