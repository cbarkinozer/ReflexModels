# Final release audit — 2026-10-15

Recommendation: ready for public release. Author-selected Apache-2.0 licensing has been applied and verified on both repositories. Dataset first, then model is appropriate. Neither repository was made public during this audit.

Verified dataset: 781 train / 90 validation / 90 calibration / 180 test, total 1,141. All downloaded export hashes match the manifest; valid request/schema/target fields; nine routes; all six split pairs have disjoint IDs and normalized request text. Hugging Face datasets loads the four JSONL splits successfully. Historical 88-case test excluded. Dataset card describes the English mobile intent-classification task and retains single-reviewer/synthetic limitations.

Verified model: adapter, joint head, configs, tokenizer and two GGUF artifacts match original verified checksums; remote binary sizes/LFS hashes agree. Correct 779,717,956 deployed text parameter count (780M), required special Decision API runtime documented, base revision pinned. No new device or GPU inference was run; functional readiness relies on the completed A32/PyTorch evaluations. Standalone merged PyTorch weights are not included and the adapter requires its base model.

Cards: metadata validation passes; Markdown table widths consistent; internal file targets present; dataset/model reciprocal links present. Main comparison remains base/Reflex/Laya on four cohorts, with hardware/profile/training-exposure limits disclosed. Credential-pattern scan found no matching HF/W&B keys or private-key blocks in audited text files; this is not a guarantee against every form of private information.

Fixed: obsolete README checksum removed from original-artifact manifest, pending GGUF mobile status corrected, private-only release wording reduced. Model and dataset revisions/verified writes are in hf-final-audit-update-20261015.json.

Licensing completed: both cards declare Apache-2.0, both repositories include the standard LICENSE and attribution NOTICE, and dataset manifest licensing is updated. Upstream training sources retain their original licenses. Verified writes are in hf-license-update-20261015.json.

Make the dataset public first, then the model. Check both pages and dataset viewer while logged out. Published test is a reproducible regression benchmark; future test-guided development needs additional unseen evidence for new generalization claims.
