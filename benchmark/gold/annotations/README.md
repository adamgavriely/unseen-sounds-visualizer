# Labels

`gold_AG.json`: every human per-sound label of the benchmark (one annotator). Per clip: `clip`, `tag`
(on-screen / off-screen category), `sounds[]` with `label`, `family`, `visible`, `obvious`, `importance`, `masked`,
`start`, `end`, plus provenance flags (`from_detector`, `from_human_label`, `suggested`, `gate`). The `sets` block
records how the file was assembled.
