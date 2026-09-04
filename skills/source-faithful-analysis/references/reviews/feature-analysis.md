# Concrete software feature review

The reviewer is never the writer. First apply `../page-review-rubric.md`, then reject the feature
page when any one of these six checks fails:

- The introduction describes implementation before establishing the problem and solution shape.
- A primitive example is presented as the full feature without proving its system placement.
- A module, variant, or companion mechanism says what it does but omits why, completion/output,
  incremental cost, or boundary where those questions apply.
- A triggered lifecycle stops before its required loss/backward handoff or the completion/visibility
  boundary defined by `../codebase.md`.
- Classes, figures, code excerpts, or an ASCII caller tree substitute for causal prose.
- The aggregate cost story contradicts or ignores costs already identified per component.

Return the base-rubric result, `feature: pass` or `feature: FAIL <smallest unit>`, and an overall
`PASS` or `REJECT`. Report findings only; the reviewer does not edit the page.
