# Repository Software Architecture Analysis Profile

## Core principle

Build one coherent explanation on a **single classification axis**: software responsibility,
dependency direction, state ownership, and interface contracts. An architecture analysis is not a
feature list, directory tour, or enlarged training/request flow. Keep the static capability view,
dynamic implementation view, source caller tree, and physical code map separate so each answers one
question precisely.

Use this profile only for an approved or boundary-defined repository-level architecture overview or
an existing architecture chapter. A new whole codebase or unplanned multi-page domain first goes to
`planning-codebase-analysis`.

## Mandatory page contract

Use this semantic order. Headings and numbering may follow the host Wiki, but every block has one
job and later blocks reuse the layer/module names established by the static view.

| Block | Reader question | Required output |
|---|---|---|
| **1. Background, goals, and scope** | Why does this software exist, and what can the frozen version do? | Design background; workload and engineering pressures; goals/non-goals; current supported, optional, experimental, legacy, external, and unsupported capability boundaries. |
| **2. Architecture and implementation** | How is the software divided, and how do the parts cooperate? | One static capability view with layer/module responsibilities, followed by a separate dynamic implementation view for one representative lifecycle and its data/control/completion interactions. |
| **3. Module design** | Why does each module exist and how is it built? | One subsection per module from Block 2: responsibility/contract, design pressure and rejected alternative, internal logic or relationship diagram, source route, and constraints/limitations. |
| **4. Architecture-to-code map** | Where is each architectural responsibility implemented? | A many-to-many mapping from the same layer/module names to directories and qualified symbols. |
| **5. Usage-scenario inventory** | How is each top-level supported scenario actually started and completed? | An evidence-derived inventory; each scenario carries an execution entry, command template, source-faithful call tree, software logic flow, completion/output, and constraints. |
| **6. Summary and reading handoff** | What should the reader now know, and where should they go next? | The system thesis, layer boundaries, representative lifecycle, scenario choice points, known limits, and links to authoritative mechanism pages. |

The default order is explanatory, not a license to create one oversized page. When the approved
blueprint splits these blocks across pages, the architecture overview keeps the compact static map,
representative lifecycle, scenario index, and links to the authoritative owners.

## 1. Establish the software boundary

- State the design background before naming packages or classes: workload scale, bottlenecks,
  integration constraints, and the naive/previous approach.
- Explain design principles as responses to those pressures. Name the alternative and deciding
  criterion; mark reconstructed rationale as analyst inference.
- Describe the current implementation's capability range at the frozen baseline. Separate core,
  optional integration, experimental, compatibility/legacy, external, and unsupported scope.

This opening is a boundary contract, not marketing history. Every claimed capability needs a live
entry, selection site, test, or published contract at the declared baseline.

## 2. Build architecture before motion

### Static capability view

Choose the classification axis before drawing. Every peer box must be the same semantic kind:
software layers grouped by responsibility and dependency direction, with modules placed under the
layer that owns their contract/state. Put cross-cutting services, extension adapters, experimental
paths, and external systems in visibly distinct regions rather than mixing them with core layers.

The static diagram and its companion table jointly state, for every layer/module:

- responsibility and functional position;
- input/output contract and dependency direction;
- state/invariant/policy ownership;
- delegated work and explicit non-responsibilities;
- supported/optional/experimental boundary; and
- load-bearing `path::qualified.symbol` evidence.

A directory name may label evidence inside a box, but directories do not define the layer model.
Feature categories may be shown as capabilities supplied by modules, not as peer “layers.”

### Dynamic implementation view

After the static map, select one real lifecycle that crosses its boundaries. Use a sequence or logic
diagram to show data, control, state ownership, synchronization, and completion/visibility signals.
Reuse the exact layer/module names from the static view. The diagram explains collaboration; it does
not replace the source execution trace required below.

## 3. Carry the partition into module design

Give every module established in the static view a basic design subsection with these slots:

1. **Responsibility and contract:** inputs, outputs, owned state, invariants, non-responsibilities.
2. **Design pressure and choice:** why this module/boundary exists, the obvious alternative, and the
   criterion that rejected it.
3. **Internal implementation:** a class/relationship/logic/sequence diagram chosen for the question,
   plus a compact stable-symbol source route.
4. **Constraints and limits:** guards, costs, unsupported combinations, fallback, failure, and
   observability.

Do not rename or regroup modules here. A discovered boundary mismatch is blueprint drift and returns
to `planning-codebase-analysis`.

### Algorithmic modules still require a principle figure

If a module or usage scenario explains an **algorithmic implementation**—a non-identity
partitioning, routing, packing, scheduling, reduction, optimization, placement, masking,
permutation, or iterative state transition—replay its smallest named input through the decisive
intermediate layout/state/owner to output. Include **at least one principle figure** that makes the
transformation, invariant, constraint, and cost reconstructable. The architecture static view,
ownership inventory, caller tree, code excerpt, prose, or table cannot substitute for that figure.

Distinct live algorithms or data planes must reuse the **same concrete example** and remain
separately traceable through local work, data/state/ownership movement, communication or
synchronization, reconstruction, applicable forward/backward differences, and incremental cost.
Use `drawing-wiki-figures` for medium choice, the rendered artifact, and stranger-reader review.
Ordinary CRUD, direct assignment, validation, and one-to-one forwarding remain exempt unless the
unit actually contains an algorithmic transformation.

## 4. Make source call relationships exact

Put every function-call relationship in a **fenced `text` ASCII caller tree**, never as an inline
`A → B → C` prose chain. This architecture contract owns the tree's presentation and module
mapping. For semantic-hop selection, direct versus transitive edges, execution-boundary annotations,
and completion semantics, follow the **Execution trace contract** in `../codebase.md`.

- Functions invoked in order by the same owner are **sequential siblings**, not children of the
  function that happened to run immediately before them.
- If a callee returns and its caller then merges, publishes, saves, or waits, show that later action
  back under the caller.
- Map each caller-tree subtree and each boundary annotation back to its **owning architecture
  module**, using the exact layer/module names introduced by the static view.

ASCII is the source-navigation medium here. Use the active figure skill for architecture, sequence,
and software-logic visuals; an ASCII caller tree does not substitute for those diagrams.

## 5. Map architecture to physical code

The architecture-to-code map is explicitly many-to-many. Use columns equivalent to:

| Layer/module from the static view | Responsibility | Directories/files | Load-bearing symbols | Boundary note |
|---|---|---|---|---|

One directory may host several responsibilities, and one architectural module may span entry,
state, backend, integration, and test directories. Record both directions rather than pretending
the repository tree is the software architecture.

## 6. Specify every usage scenario

Derive the usage-scenario inventory from live entry scripts/modules, launch wrappers, examples,
public docs, configs, tests, and tools. Classify core, optional, experimental, legacy, and broken or
stale documented routes. “All scenarios” means every top-level way the frozen repository presents
itself as executable, not every internal function variant.

Each scenario uses this fixed card:

1. **Execution entry:** the concrete script/module/launcher, required services, config/model/data
   inputs, hardware/process assumptions, and expected artifact or response.
2. **Command template:** a fenced shell block with the real launcher and script/module. Mark user
   placeholders; do not invent defaults that the source does not provide.
3. **Function calls:** a fenced `text` ASCII caller tree from entry to completion.
4. **Software logic:** a separate flow/sequence diagram for branches, data movement, and boundaries.
5. **Completion and limits:** the completion boundary, observable output, guards, unsupported cases,
   and known failure path.

Before publishing a command, **verify the script** or module exists at the frozen commit and
**verify every flag** against the parser, wrapper, or delegated parser that consumes it. Verify
launcher/environment variables and prerequisites too. When documentation and implementation
disagree, report both; use current implementation for “what happens now” and identify the stale or
broken documented route. If a shipped entrypoint fails before useful work, show the verified failure
call path. Offer a workaround only when its source path is valid, and distinguish source validation
from runtime validation on unavailable hardware/services.

## Sizing and ownership gate

When the host maintenance skill's page-split signal is reached, the planner decides whether to
separate the architecture overview, module deep dives, and usage/quickstart page. A writer does not
split an approved page locally; it reports the proposed ownership change and evidence to
`planning-codebase-analysis`.

## Completion gate

An **independent reviewer** who was never the writer applies
`../reviews/software-architecture.md`. Completion requires:

- one coherent classification axis and exact module-name carry-through;
- visibly separate static, dynamic, caller-tree, and directory-map views;
- source-verified commands and scenario coverage, including contradictions and broken paths;
- ASCII layout that preserves sibling/return ownership, maps subtrees to modules, and passes the
  source pack's execution-trace checks; and
- explicit completion, output, constraints, failure semantics, and reader handoff.
