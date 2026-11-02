# Model cards of the context-shadow challengers

One immutable JSON card per FROZEN version, named `MODEL_CARD_<model_version>.json`, with: model_version,
role, lineage, changes, must_not_contain, inputs, per-component and combined implementation hashes (sha256
of the files that define the mathematics), pinned commit, environment, governance pointers, card_sha256.
`shadow_context/VERSIONS.json` records the card's sha256; `hoopslab.pregame.versions.verify_version` refuses
a version whose card is missing, altered or PLANNED. The baseline's card stays in `shadow/`.

No card exists here yet (2026-11-02): v0.2, v0.3 and v0.4 are PLANNED and have no code.
