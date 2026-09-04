# Concrete software feature review

The reviewer is never the writer. First apply `../page-review-rubric.md`, then reject the feature
page when any one of these checks fails:

- The introduction describes implementation before establishing the problem and solution shape.
- A primitive example is presented as the full feature without proving its system placement.
- The smallest example cannot be replayed from named input identity/shape through each decisive
  intermediate state, layout, or owner to its merge/output.
- A module, variant, or companion mechanism says what it does but omits why, completion/output,
  incremental cost, or boundary where those questions apply.
- A distinct live variant or data plane is not replayed with the same concrete example through local
  compute, communication or synchronization, reconstruction, applicable backward difference, and
  incremental cost; or that path is missing from the rendered principle figure required by the base
  algorithm-replay gate.
- The variant set is asserted without an enumeration basis drawn from the source's own selection
  sites, or a sibling selection axis covering the same concern for another entity class is neither
  explained nor named with its owner.
- Behavior inside a third-party dependency is narrated as verified execution rather than as its
  published contract, or the page does not say what the analyzed source proves versus what it
  cannot at that boundary.
- The page replaces an earlier version and something it previously owned—an evidenced claim, a
  correction, a registered configuration or interface name, a relied-upon cross-link—is neither
  present, nor corrected with evidence, nor rehomed to a named owner.
- A triggered lifecycle stops before its required loss/backward handoff or the completion/visibility
  boundary defined by `../codebase.md`.
- Classes, figures, code excerpts, or an ASCII caller tree substitute for causal prose.
- The aggregate cost story contradicts or ignores costs already identified per component.

Return the base-rubric result, `feature: pass` or `feature: FAIL <smallest unit>`, and an overall
`PASS` or `REJECT`. Report findings only; the reviewer does not edit the page.
