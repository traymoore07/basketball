# Data-contract mismatch log

A **DATA-CONTRACT MISMATCH** is a difference between the frozen data contract (`src/hoopslab/contract/data_contract.yaml`, `docs/prereg/05`, `docs/prereg/11`) and what the real corpus actually provides. Examples:

- a required field does not exist;
- a knowledge time cannot be established;
- coverage is too low for a level;
- an identifier is not opaque;
- a table exists only for some leagues or seasons.

Recording a mismatch **does not change the pre-registration.** A mismatch has exactly one of these consequences, chosen by the rule in `docs/onboarding/00_onboarding_protocol.md` §3:

| Consequence | Meaning | Needs an amendment? |
|---|---|---|
| `NONE` | cosmetic (naming, units, encoding); resolved inside the adapter | no |
| `ADAPTER-FALLBACK` | resolved by a pre-registered fallback rule (e.g. conservative knowledge-time imputation, contract §1) | no; the LEAKAGE-SENSITIVE rerun still applies |
| `LEVEL-DOWNGRADE` | the experiment runs at the lower data level the frozen experiment-to-level map already allows | no |
| `SCOPE-RESTRICTION` | the experiment runs as frozen but only on the leagues/seasons where the data exists | no, but reported next to every affected result |
| `BLOCKED` | the frozen experiment cannot run at all; it is reported as **NOT EXECUTABLE (M<n>)**, never silently dropped | no |
| `PROXY-REQUIRED` | running it would need a substitute variable, definition, or metric not in the frozen plan | **yes**: an amendment, class DATA-DRIVEN NECESSITY; the original is also reported as NOT EXECUTABLE |

Fields:

- **Contract reference:** table.field, or the checklist item in `docs/prereg/11`.
- **Evidence:** a structural fact only. Schema, presence or absence, null share, coverage counts, provenance. It must not involve outcome statistics relevant to any hypothesis (see the exposure rules in the protocol §2).

| ID | Date (UTC) | Contract reference | Expected (frozen) | Found in corpus | Evidence (structural only) | Leagues / seasons | Experiments / hypotheses affected | Consequence | Amendment |
|---|---|---|---|---|---|---|---|---|---|

*(no mismatches recorded; corpus not yet inspected)*
