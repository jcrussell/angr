You are continuing autonomous work on the **rust-symex optimization epic** in
`{{.RepoRoot}}`.

- Iteration: **{{.Iter}}**
- State: `{{.State}}` (previous: `{{if .PrevState}}{{.PrevState}}{{else}}none{{end}}`)
- Git HEAD: `{{.GitHead}}` (dirty: {{.GitDirty}})
{{if .GateResult}}- Previous gate result: **{{.GateResult}}**
{{end}}
The session-handoff note from the previous iteration is at
`.ralph/state/session.md` — read it first to recover context.
