# Software architecture review rubric

## Who runs it

- The reviewer is never the writer. The reviewer reports findings and does not edit the page.
- Input is the finished architecture page, its approved page contract, and read access to the frozen
  source checkout.
- First apply the five base checks in `../page-review-rubric.md`; then apply all eight checks below.
- A failed item names the smallest section/scenario and returns to the writer. Material ownership
  drift returns to `planning-codebase-analysis` instead of being repaired during review.

## Eight architecture checks

1. The design background states context, goals and non-goals, the current capability boundary, and
   supported/unsupported scope. One responsibility/dependency/state-ownership classification axis
   is used; peer layers are not a mix of features, directories, deployment roles, and software
   layers.
2. Static capability, dynamic implementation, and physical code-map views reuse the same
   layer/module names. The ASCII caller tree uses real qualified symbols and maps each subtree and
   execution-boundary annotation back to its owning module.
3. Every module in the static view has responsibility/contract, design rationale and rejected
   alternative, internal implementation evidence, and constraints/limitations.
4. The architecture-to-code mapping is many-to-many and anchored by qualified symbols.
5. The top-level usage-scenario inventory reconciles live scripts/modules, examples, docs, tools,
   tests, and explicitly classified optional/experimental/legacy/broken routes.
6. Spot-checked commands name an existing frozen-baseline entry and every flag resolves to its real
   parser/wrapper; documentation/implementation conflicts remain visible.
7. ASCII call trees pass the source-faithful execution-trace contract plus the architecture
   presentation checks: exact-parentage hop-walk, sequential siblings, returns to callers, and
   subtree/boundary mapping to the owning module.
8. Every scenario states prerequisites, observable output, completion boundary, constraints, and
   failure semantics. The final reader handoff states what the reader now knows, the remaining
   choice points and limitations, and the authoritative next pages.

## Verdict

Record the five base-check results, `architecture: pass` or
`architecture: FAIL <smallest item/scenario>`, the three spot-checked anchors/commands, and one
overall `PASS` or `REJECT`. Any failed base or architecture check rejects the page.
