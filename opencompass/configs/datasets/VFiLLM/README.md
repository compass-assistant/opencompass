# V-FiLLM 10-Q Configuration

Place the official `random_questions_10q_d1-5.csv` and `atoms_10q.json`
repository artifacts under `data/V-FiLLM/`, or set both dataset path fields in
the configuration to their operator-provided locations. The selected benchmark
scope evaluates the first 250 mixed-depth questions, uses the official 10-Q
prompt renderer, and applies the deterministic relative-tolerance evaluator.

The frozen source attachment identifies these files by repository path but does
not provide a public archive URL or content revision. Operators must obtain the
matching artifacts from the official V-FiLLM repository before execution.
