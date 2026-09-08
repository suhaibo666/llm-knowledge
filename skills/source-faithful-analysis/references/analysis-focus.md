# Choose the explanation that fits the subject

Read this with the selected document profile. The profile defines the deliverable; the subject and
reader question determine what needs explaining inside it. These are investigation lenses, not
additional page types or mandatory section templates.

## Select the decisive mechanism

First identify what the reader must be able to reconstruct or decide, and which source operation
establishes that result. Choose a primary lens from the evidence; combine lenses only where the
actual mechanism crosses them.

| Observable subject | Explanation to develop | Evidence to seek |
|---|---|---|
| Algorithm or data structure | Input representation; decisive rule, recurrence, or update; worked example; correctness and termination; time/space complexity | Implementation branches and loop bounds, proof or derivation, edge-case tests |
| Numerical method or compute kernel | Equations, dtype and layout transformations, rounding/error behavior, execution organization, compute/memory costs | Arithmetic and indexing, dispatch conditions, numerical tests, measured or derived cost |
| Stateful, concurrent, or distributed process | State and resource ownership, transitions, communication, ordering, completion/visibility, partial failure and recovery | State mutations, queues/RPCs, waits/fences, publication and cleanup, concurrency tests |
| Interface, parser, registry, or adapter | Accepted input, selection and transformation rules, composition, compatibility, output/error contract | Parser/schema, factory/registry, adapters, supported-value and negative tests |
| Repository architecture | Responsibility and dependency structure, interfaces, representative collaboration, capability and scenario boundaries | Construction and integration sites, module contracts, live entrypoints and scenarios |
| Empirical paper, dataset, or incident | Hypothesis/question, method or causal account, observations, comparison basis, uncertainty and limitations | Sections/tables, records and filters, event timelines, competing explanations |

For example, binary search is explained through interval bounds, duplicate handling, shrinking
search space, and termination. Its local loop variables do not justify a service-ownership table.
Quantization is explained through scale, integer mapping, reconstruction and error; persistent
ownership becomes relevant only if the page also owns loading, caching, or publication. A cache
refresh service does need ownership and visibility analysis when background jobs race with readers.

## Apply depth at the point where it is needed

- For a pure function or local algorithm, the returned value and its properties close the path.
  Follow the decisive arithmetic, branch, recurrence, or memory access even if it is all inside one
  function. A caller tree cannot substitute for explaining those operations.
- For a multi-module feature, explain the primitive and then its actual consumers and integration.
  For a standalone algorithm, the public function and its input/output contract may be the entire
  integration boundary; no artificial service lifecycle is needed.
- For persistent/shared state, asynchronous work, resource transfer, or externally visible effects,
  apply the codebase pack's ownership, execution, completion and failure questions. Retain that
  depth for the affected path even when another part of the page is purely numerical.
- For a design choice, name a concrete alternative and the deciding criterion. For an individual
  mathematical or mechanical step, explain why the rule is valid and necessary. Do not invent a
  rejected product architecture for every helper or formula; label reconstructed rationale.
- For constraints, distinguish explicit guards from algorithmic preconditions, numerical limits,
  resource costs, and unsupported cases. Identify the actual enforcing code/test or state that no
  runtime guard exists; absence of an assertion is not absence of a boundary.

## Make depth accessible

Establish the problem and enough vocabulary to follow a smallest meaningful example. Use that
example to explain the decisive operation before expanding to detailed implementation, variants,
and uncommon cases. For comparable variants, reuse its input so their differences can be assessed.
Use the selected profile's principle-figure trigger and the host's medium rules; this focus choice
does not waive a required figure or turn every example into a distributed layout.

Explain domain terms at first use or link a focused prerequisite. Place a link to the authoritative
page where the next question arises, with a meaningful label and purpose. Keep implementation
evidence navigable without making readers decode a file/field inventory before meeting the idea.

Review success means that a reader can reconstruct the mechanism and continue into its evidence.
An otherwise complete algorithm page is not missing an ownership table, publication stage, or RPC
trace when the source has none. A concurrent page is incomplete if those relevant guarantees are
left unexplained. Review the selected lens against the actual source, not keyword counts.
