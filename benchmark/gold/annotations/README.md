# Labels

`gold_AG.json`: every human per-sound label of the benchmark (one annotator). Per clip: `clip`, `tag`
(on-screen / off-screen category), `sounds[]` with `label`, `family`, `visible`, `obvious`, `importance`, `masked`,
`start`, `end`, plus provenance flags (`from_detector`, `from_human_label`, `suggested`, `gate`). The `sets` block
names the second batch.

`gold_AG_v1.3.10.json`: the same labels before the five test-label corrections below (as in release v1.3.10).

## Test-label corrections

After the test outputs had been scored, the test set was re-listened to blind: 92 questions at 76 moments in 51 test
clips, built from the outputs of four systems (final system, direct audio-to-image, the two Qwen3-Omni baselines) with
the system names hidden, covering needed sounds that at most one system found and wrong pictures of the final system,
of direct audio-to-image, or of two or more systems. Only answers confirmed on a second listen were applied. Times of
added sounds were measured from the audio. Every system was re-scored from its saved outputs on the corrected labels
(report, Section 9 and Appendix E.1).

| clip | change |
|---|---|
| `tg_d009` | added: laughter (person filming), 8.53–9.25 s, off screen, importance 2 |
| `m4_airsoft_24a` | added: laughter, 3.2–4.3 s, off screen, importance 2 |
| `tg_d078` | removed: vehicle horn at 5.2 s (not in the audio) |
| `ly_alarm_qCVdVav` | renamed: electric toothbrush → buzzer (5.2–18.0 s) |
| `w8_helmetcam_chainsaw_roof_2b` | renamed: chainsaw → motorcycle (10.3–16.8 s) |
