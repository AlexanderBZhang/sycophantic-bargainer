# Roadmap: From Prototype to Valid Research

This document tracks the required fixes and validation steps before this framework can produce valid experimental data.

---

## Phase 1: Critical Bug Fixes

**Goal**: Resolve all bugs that invalidate data collection.

### 1.1 Premature Termination Bug

**Problem**: When LLM moves first, the game loop interprets opening proposals as acceptances.

**Tasks**:
- [ ] Implement explicit turn-state machine with states: `AWAITING_FIRST_MOVE`, `AWAITING_RESPONSE`, `ROUND_COMPLETE`
- [ ] First message from each agent must be classified as "proposal" not "acceptance"
- [ ] Add unit test: first-mover proposal should not terminate negotiation

**Owner**: TBD  
**Estimated effort**: 2-4 hours

### 1.2 Round Counting Bug

**Problem**: Game loop allows round 11 when MAX_ROUNDS = 10.

**Tasks**:
- [ ] Move round check to BEFORE soliciting agent response
- [ ] Add unit test: negotiation terminates at exactly MAX_ROUNDS
- [ ] Review all loop termination conditions

**Owner**: TBD  
**Estimated effort**: 1-2 hours

### 1.3 Hardball Repetition

**Problem**: Hardball sends identical messages repeatedly, which LLMs recognize.

**Tasks**:
- [ ] Implement escalation logic (instead of repeating, increase pressure or walk)
- [ ] Add unit test: Hardball never sends identical consecutive messages

**Owner**: TBD  
**Estimated effort**: 2-3 hours

### 1.4 Deterministic Termination Exploit

**Problem**: Known MAX_ROUNDS enables trivial accept anything > BATNA on final round strategy.

**Options**:
- [ ] **Option A**: Breakdown probability (p% chance of termination per round)
- [ ] **Option B**: Discount factor (surplus decreases by δ each round)
- [ ] **Option C**: Hidden horizon (don't reveal MAX_ROUNDS to agents)

**Recommendation**: Option A (breakdown probability) most closely models real negotiation uncertainty.

**Tasks**:
- [ ] Implement chosen mechanism
- [ ] Update utility calculations to account for new dynamics
- [ ] Add unit test: agents cannot perfectly predict final round

**Owner**: TBD  
**Estimated effort**: 3-5 hours

### 1.5 Parser Information Leak (Low Priority)

**Problem**: Hardball receives full message text, not just bracketed actions.

**Impact**: ⚠️ **Design note, not logical error.** Hardball is rule-based and only extracts numerical values via regex—it ignores natural language entirely. No decision-making is affected.

**When it would matter**:
- If Hardball were replaced with an LLM that could be influenced by persuasive language
- If testing whether structured-only protocols change LLM behavior

**Tasks** (optional cleanup):
- [ ] Modify parser to extract only `[[ACTION: params]]` content before passing to Hardball
- [ ] Add unit test: Hardball receives only structured actions

**Owner**: TBD  
**Estimated effort**: 1-2 hours  
**Priority**: Low (does not invalidate data)

---

## Phase 2: Test Coverage

**Goal**: Ensure game logic is correct before data collection.

### 2.1 Unit Tests for Game Loop

- [ ] Test: Turn alternation is correct
- [ ] Test: Payoffs computed correctly for all outcomes (agreement, walk-away, timeout)
- [ ] Test: Round counting is accurate
- [ ] Test: BATNA applied when negotiation fails

### 2.2 Integration Tests

- [ ] Test: Full negotiation completes without errors (mock LLM)
- [ ] Test: Transcript contains all expected fields
- [ ] Test: Language analysis pipeline runs on valid transcript

### 2.3 Manual Verification

- [ ] Review 10+ full transcripts for protocol adherence
- [ ] Verify payoff calculations match transcript content
- [ ] Check that agent turns align with expected structure

### 2.4 Repository Cleanup

**Problem**: Multiple scripts with overlapping functionality exist in `scripts/`.

**Potential duplicates to audit**:
- Tournament scripts: `run_h2h_tournament.py`, `batch_tournament.py`, `hybrid_batch_tournament.py`, `run_realtime_tournament.py`
- Hardball scripts: `run_hardball_experiment.py`, `run_hardball_batch.py`
- Parsing scripts: `parse_enron_baseline.py`, `parse_enron_fast.py`

**Tasks**:
- [ ] Audit scripts for overlapping functionality
- [ ] Consolidate or remove duplicates
- [ ] Update imports and references
- [ ] Document canonical scripts in README

---

## Phase 3: Methodology Validation

**Goal**: Ensure metrics measure what we claim they measure.

### 3.1 Sycophancy Marker Validation

- [ ] Collect human and LLM ratings of sycophancy for sample messages
- [ ] Compute correlation between marker counts and ratings
- [ ] Compute inter-rater reliability (Krippendorff's alpha ≥ 0.7)
- [ ] Document marker set with validation evidence

### 3.2 Utility Function Validation

- [ ] Verify sigmoid parameters produce expected utility landmarks
- [ ] Test penalty zone: confirm sub-$35M deals are penalized
- [ ] Document rationale for parameter choices

### 3.3 Baseline Validation

- [ ] Use CaSiNo (Campsite Negotiation Dialogues) as primary human baseline
- [ ] Document CaSiNo limitations (campsite resources vs. money, cooperative framing)
- [ ] Use Enron corpus as secondary baseline for general business communication
- [ ] Acknowledge context differences in all outputs

---

## Phase 4: Pre-Registration

**Goal**: Commit to hypotheses before data collection.

### 4.1 Pre-Registration Document

- [ ] State primary hypothesis (P1: Sycophancy correlates with surplus loss)
- [ ] State secondary hypotheses (P2-P4)
- [ ] Specify primary statistical test (Mann-Whitney U)
- [ ] Specify alpha level (0.05) and correction for multiple comparisons
- [ ] Specify minimum sample size (N per condition)
- [ ] Register on OSF or AsPredicted

### 4.2 Analysis Code Lock

- [ ] Freeze analysis code before data collection
- [ ] Hash and timestamp analysis scripts
- [ ] Any post-hoc analyses must be labeled as exploratory

---

## Phase 5: Data Collection

**Goal**: Run valid experiments.

### 5.1 Pilot Study

- [ ] Run N=10 negotiations per condition
- [ ] Verify transcripts are complete and parseable
- [ ] Check for unexpected errors or edge cases
- [ ] Adjust protocol if needed (document changes)

### 5.2 Full Study

- [ ] Symmetric play: N=80 per pairing (Claude-Claude, GPT-GPT, Claude-GPT)
- [ ] Adversarial play: N=90 per condition (Standard, Thinking)
- [ ] Store raw transcripts with timestamps and model versions

### 5.3 Data Validation

- [ ] Automated checks: all negotiations have valid payoffs
- [ ] Automated checks: no negotiations exceed MAX_ROUNDS
- [ ] Manual spot-check: 5% random sample for transcript coherence

---

## Phase 6: Analysis and Reporting

**Goal**: Report findings.

### 6.1 Pre-Registered Analyses

- [ ] Run primary statistical test
- [ ] Report effect size with confidence interval
- [ ] Report power achieved

### 6.2 Exploratory Analyses

- [ ] Label all non-pre-registered analyses as exploratory
- [ ] Apply appropriate corrections for multiple comparisons
- [ ] Do not claim "significance" for exploratory findings

### 6.3 Limitations Section

- [ ] Document all remaining limitations
- [ ] Acknowledge any deviations from pre-registration
- [ ] Provide data and code for replication

---

## Status Summary

| Phase | Status | Blocking Issues |
|-------|--------|-----------------|
| Phase 1: Bug Fixes | ❌ Not Started | All bugs unresolved |
| Phase 2: Test Coverage | ❌ Not Started | Depends on Phase 1 |
| Phase 3: Methodology Validation | ❌ Not Started | Depends on Phase 2 |
| Phase 4: Pre-Registration | ❌ Not Started | Depends on Phase 3 |
| Phase 5: Data Collection | ❌ Not Started | Depends on Phase 4 |
| Phase 6: Analysis | ❌ Not Started | Depends on Phase 5 |

---

## Timeline (Estimated)

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1 | 1-2 weeks | 1-2 weeks |
| Phase 2 | 1 week | 2-3 weeks |
| Phase 3 | 1-2 weeks | 3-5 weeks |
| Phase 4 | 1 week | 4-6 weeks |
| Phase 5 | 2-3 weeks | 6-9 weeks |
| Phase 6 | 1-2 weeks | 7-11 weeks |

**Total**: 2-3 months to produce valid, pre-registered findings.

---

*Last updated: January 2026*
