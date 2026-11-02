# Structure Hold: assurance pack (parts list, origin, standards, sign-off)

> **Confidential. Copyright (c) 2026 Zain Ali. All rights reserved.** For a drone maker
> who has signed an evaluation or licence agreement with us, and nobody else.

| | |
| --- | --- |
| Document | Structure Hold assurance pack |
| Version | 0.1, draft: the fields in section 3 and the safety manual signature are still to fill |
| Date | 2026-11-02 |
| Checked against | `main` at `e185a80` (policy v22, PX4 v1.17.0) |
| Owner | Zain |
| Companions | `docs/safety-manual.md` (hazards, failure table, sign-off), `docs/integration-guide.md` (the interface) |

This is the paperwork a buyer's procurement, security and safety reviewers ask for before
an evaluation: what the software is made of, where it came from, how it lines up against
the standards they work to, and who has signed for it. It comes from the "assurance pack"
item of the buyer research of 2026-11-02 (government and utility customers of drone makers
ask for it first).

**What we do not claim.** Structure Hold holds no certification, approval or listing of any
kind: no type certificate, no ASTM or ISO conformity statement, no Blue UAS or Green UAS
listing, no DO-178C assurance level. Every result is from simulation and unit tests; the
mode has not flown on a real drone. The tables below say what we do and what is missing,
standard by standard, so a reviewer can judge the gap rather than take our word for it.

## 1. What is in the pack

| Item | Where | Status |
| --- | --- | --- |
| Software parts list (SBOM), CycloneDX 1.6 JSON | `sim2real/assurance/sbom.cdx.json` | Generated from the build files, checked by CI |
| Statement of origin | Section 3 below | Written; company fields for Zain to fill |
| Standards mapping | Section 4 below | Written from the standards' published scopes |
| Safety manual and failure table (FMEA) | `docs/safety-manual.md`, sign-off in its section 8 | v1.0 ready, **not signed yet** |
| Interface spec | `docs/integration-guide.md` | Current, pinned to the module by tests |
| How to report a vulnerability | `.github/SECURITY.md` | In place |
| Flight test plan for a partner's drone | `docs/partner-pack.md` | Ready, no flight yet |

## 2. Software parts list (SBOM)

The parts list is a CycloneDX 1.6 file, the format most supply-chain tools read. Each part
carries its supplier, licence, version (or the version range we declare) and a
`dronenav:role` saying where it runs:

| Role | Parts | Where they run |
| --- | --- | --- |
| `onboard` | Structure Hold module, the compiled-in policy, PX4-Autopilot v1.17.0 | In the drone's flight controller firmware |
| `evaluation` | Our Python tools, NumPy, PyYAML, pyulog, pymavlink, MAVSDK-Python, Gazebo Harmonic, Ubuntu base image | Desktop or server: SITL image and scoring |
| `training` | PyTorch, rsl-rl-lib 5.5.1, tensordict, ONNX, ONNX Runtime | Desktop or server, only when training a policy |
| `development` | pytest, ruff | Tests and lint |
| `research` | Basalt VIO, rpg_trajectory_evaluation, EuRoC MAV dataset | Offline, once, to fit the simulator's estimate error; not delivered |

What a reviewer usually wants to know, in plain words:

- **On the drone there is our code and PX4, nothing else.** The module is plain C++ that
  uses only PX4's own libraries (uORB, the work queue, mathlib, mavlink_log) and the C++
  standard headers. It has no ONNX runtime, no heap allocation and no network code of its
  own; it talks only through PX4's uORB topics, and to the pilot through PX4's normal
  MAVLink messages. A test (`sim2real/assurance/test_sbom.py`) fails if the module ever
  includes or links anything else without the parts list changing.
- **No copyleft code goes on the drone.** PX4 is BSD-3-Clause. The only copyleft part
  anywhere is pymavlink (LGPL-3.0), which runs on the desktop side of the test tools and is
  never linked into the module.
- **PX4's own parts are the drone maker's.** The maker builds and ships PX4 (with NuttX and
  its submodules), so those belong in the maker's firmware SBOM; ours names PX4 v1.17.0 as
  the host.

The checked-in file lists the declared parts only, so it changes only when a part changes.
For a delivery, make a stamped copy, which adds the date, the git commit, the SHA-256 of
every file that goes into the firmware, and the exact versions installed:

```
python -m sim2real.assurance.make_sbom --release runs/sbom_<date>.cdx.json
```

After any change to dependencies, the module's includes or the released policy, run
`python -m sim2real.assurance.make_sbom` and commit the result; CI fails until you do.

## 3. Statement of origin

To fill before this goes to anyone:

| Field | Value |
| --- | --- |
| Legal owner of the software | Zain Ali (copyright holder, `LICENSE`) |
| Trading or company name | *(Zain to fill)* |
| Country of the owner | *(Zain to fill)* |
| Contact for this statement | *(Zain to fill)* |

What we state, and how a reviewer can check it:

1. **Who wrote it.** The Structure Hold module, the policy and the tools in this repository
   were written for this project under the direction of Zain Ali, the copyright holder,
   who reviews and merges every change. Software tools used in writing it include AI coding
   assistants. No code was bought, licensed in or taken from another proprietary product.
   Every commit on `main` is authored by the owner's account, every change went through a
   pull request with automated tests, and the history is kept.
2. **Third-party code.** All of it is open source under the licences in the parts list
   (section 2), used unmodified except Basalt, which got one small patch for the offline
   research run (`eval/vio_basalt/`). None of it comes from a company we know to be on a
   US covered-foreign-entity list; the suppliers are named in the parts list so a buyer's
   counsel can check that themselves.
3. **The policy weights.** The learned controller was trained by us, from scratch or from
   our own earlier releases, with PPO on our own simulator (`envs/`). Its training data is
   simulated only: no recorded flights, no customer data, no personal data, no third-party
   or pre-trained weights. Every training run is logged (`experiments/LOG.md`) with its
   commit and config, and each release says which runs made it
   (`training/policies/README.md`).
4. **Public data.** The simulator's position-error model was fitted to real VIO error
   measured on the public EuRoC MAV dataset (`docs/vio-fit.md`). Only the fitted numbers are
   kept; the dataset is not in the repository, not in the policy and not redistributed.
5. **Hardware.** None. Structure Hold is software only; it runs on the drone maker's own
   flight controller.
6. **Where it was made.** Source is held in one private GitHub repository. Development and
   simulation ran on the owner's own computer and on cloud development machines; no part of
   the software runs on, or calls, any outside service when it flies.

This statement gives facts for a buyer's own compliance checks (for example the US NDAA
section 848 and American Security Drone Act rules on covered foreign entities). It is not
a legal determination of compliance with any of them.

## 4. Standards mapping

Mapped from each standard's published scope and summary, not from a clause-by-clause
reading: ASTM and other standards bodies sell the full texts, and we have not bought them.
A clause-level mapping for ASTM F3201 and F3269 is the next step once a buyer asks for it
(section 6). "Partly" means we do some of what the standard is about, not that we conform.

| Standard | What it is about | What we have | Gap |
| --- | --- | --- | --- |
| [ASTM F3201-26](https://store.astm.org/f3201-26.html), dependability of software used in UAS (replaces F3201-24) | Process, not performance: consistent processes across the software life, configuration management of every release, and verification and validation testing suited to the intended use (simulation, SITL, HIL or flight, the provider's choice); an alternative to DO-178-style assurance | **Partly.** One repository, every change by pull request with CI (lint, Python tests on three Python versions, the module's C++ tests, and a SITL hover flight when `sim2real/` changes); releases and runs logged with commit and config; V&V by Monte Carlo mock simulation, PX4 SIH and Gazebo SITL against tuned PX4 (`docs/baseline.md`, `docs/field-faults.md`) | No written software development plan or release procedure in F3201's terms; no HIL or flight testing yet; no clause-level check |
| [ASTM F3269-21](https://store.astm.org/f3269-21.html), run-time assurance (RTA) to bound complex functions | An architecture where a simple, assured monitor bounds a complex or unassured function (such as a learned controller) and switches to a recovery function at safe boundaries | **Partly; this is how the module is built.** The learned policy is bounded by a classical watchdog (`Watchdog.hpp`) that trips on an invalid, stale, frozen or freshly reset estimate, too much tilt, too large a hold error or a lost rangefinder, and hands back in a fixed order (`Handback.hpp`) to stock PX4 modes or the guarded Structure Hover; the trips and handback are unit-tested and listed in the safety manual's failure table | Boundaries are set from simulation, not from a safety assessment at a design assurance level; F3269 points to ARP4754A, ARP4761 and F3178 for that, which we have not done |
| NTIA minimum elements for an SBOM (2021) and CISA's guidance | Supplier, component name, version, other identifiers, dependency relationships, SBOM author and time stamp | **Yes in the stamped copy** (`--release`): all fields, plus SHA-256 of the onboard files. The checked-in copy omits the time stamp and leaves ranged Python versions as declared ranges | None for the onboard parts; PX4's own parts are in the maker's SBOM |
| NIST SP 800-218, Secure Software Development Framework | Protect the code, produce well-secured software, respond to vulnerabilities | **Partly.** Private repository, single owner, reviewed pull requests, CI on every change, pinned versions for PX4, rsl-rl, MAVSDK and the research tools, a vulnerability reporting route (`.github/SECURITY.md`), the module clamps its own parameters to their documented ranges (`ParamLimits.hpp`) | No signed releases, no automated dependency vulnerability scan, no formal threat model |
| EU Cyber Resilience Act (Regulation 2024/2847) | Security duties for products with digital elements sold in the EU, including an SBOM, vulnerability handling and reporting | Parts list and a reporting route exist | **Not assessed**: whether it applies to a component sold to a drone maker, and how it sits with EU drone rules, needs legal advice before EU sales |
| EU AI Act (Regulation 2024/1689) | Duties for AI systems, heavier where AI is a safety component of a regulated product | Learned controller bounded by a classical watchdog; training runs logged | **Not assessed**: needs legal advice before EU sales |
| EASA / JARUS SORA 2.5 (specific category), UK CAA equivalents | The operator's risk assessment for a given flight | The safety manual's envelope, do-not-use list and failure table are written to feed an operator's assessment | The assessment itself is the operator's, per mission; we supply evidence, not approval |
| RTCA DO-178C | Development assurance for airborne software in certified aircraft | Not followed | Not applicable to the target drones today; F3201 is the UAS alternative above |
| Blue UAS (US DoD) and Green UAS (AUVSI) | Listing or certification of drones and parts for government use | None | Not applied for; these list whole drones and components, and would come with a maker's drone |

Also not assessed yet: export classification of the software (drone flight control can be
controlled under dual-use export rules in the UK, EU and US). Needs a check before any sale
outside the owner's country.

## 5. Safety manual sign-off

The safety manual (`docs/safety-manual.md`) is the safety case: the operating envelope, the
do-not-use list, the safety layers, every pilot message, and the failure table with a test
behind each row. Version 1.0 is ready and **unsigned**. Zain signs it himself in its
section 8 (Approved by, owner); a partner's reviewer can add their own signature in the
same table. Signing confirms the manual matches what was built and tested at the commit it
names, and accepts its open items for simulation and a partner's supervised test flights;
it does not certify the mode for unsupervised or commercial use.

Its pilot messages and parameter defaults are checked against the module by
`sim2real/px4/tests/test_safety_manual.py`, so the signed copy cannot silently drift from
the code. Any change to its sections 2 to 6 makes a new version that is signed again.

## 6. Before sending, and next steps

1. Fill the three company fields in section 3 (Zain).
2. Sign the safety manual, section 8 (Zain).
3. Make a stamped parts list for the delivery (`--release`) and send it with this file.
4. Have a lawyer read section 3 and the EU and export rows of section 4 before the first
   EU or export sale.
5. When a buyer asks for it: buy ASTM F3201-26 and F3269-21 and do the clause-level mapping;
   add a written development plan and release procedure, which closes most of the F3201 gap.

## Change control

Bump the version and date at the top when a row changes. The parts list follows the code
automatically (CI fails until it is regenerated); sections 3 and 4 change by hand in the
same pull request as the change they describe.
