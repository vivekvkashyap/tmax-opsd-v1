// Claude Code Workflow for OPSD hint generation.
// Launch with the Workflow tool:
//   Workflow({ scriptPath: "<repo>/scripts/hintgen/hint_workflow.js",
//              args: { repo: "<abs path to tmax-opsd-v1>", count: 300 } })
// It reads batch files from <repo>/.hintgen/batches/batch_0000.json ...
// (created by slice_remaining.py) and writes <repo>/data/demos/<task_id>.md.
// Resumable: agents skip tasks whose demo file already exists.
// Cyber-flagged batches fall back to per-task singles so one bad task can't
// sink the other 11. Chunk to stay under the 1000-agent-per-workflow cap
// (count=300 -> 300 batches of 12 = ~3600 tasks per launch).

export const meta = {
  name: 'opsd-hint-generation',
  description: 'Generate method-level OPSD hints (Sonnet 5) for a chunk of remaining tmax tasks; one file per datapoint, skip done, isolate cyber-blocked tasks as singles',
  phases: [{ title: 'Generate', detail: 'Sonnet 5 subagent per 12-task batch writes data/demos/<task_id>.md' }],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const REPO = A.repo || '/path/to/tmax-opsd-v1'  // pass args.repo = absolute repo path
const count = A.count || 1
const BATCH = 12
const INSTR = `${REPO}/scripts/hintgen/hint_instructions.md`
const DEMOS = `${REPO}/data/demos`
const DIR = `${REPO}/.hintgen/batches`

const WRITTEN = { type: 'object', properties: { written: { type: 'array', items: { type: 'string' } } }, required: ['written'] }

const genPrompt = (bp) => `Read the hint-writing instructions at ${INSTR} and follow them exactly. Your batch file is ${bp} (JSON array of tasks with task_id, description, truth).
RESUMABILITY: before writing a task's hint, if ${DEMOS}/<task_id>.md already exists non-empty, SKIP it.
MAPPING (critical): write each hint to the file named by ITS OWN task_id; never put one task's hint in another's file. After writing, re-open each file and confirm it matches that task_id.
Return the task_ids you wrote or confirmed present in "written".`

const singlePrompt = (bp, i) => `Read the hint-writing instructions at ${INSTR} and follow them exactly. From the JSON array in ${bp}, process ONLY the element at index ${i} (0-based). Write that one task's hint to ${DEMOS}/<task_id>.md. If it already exists non-empty, skip. If the task trips a safety refusal, skip it. Return the task_id in "written" (empty array if nothing written).`

const batchPaths = Array.from({ length: count }, (_, i) => `${DIR}/batch_${String(i).padStart(4, '0')}.json`)
log(`hint generation: ${count} batches of ${BATCH} (~${count * BATCH} tasks) on Sonnet 5`)

const results = await pipeline(batchPaths, async (bp, _o, idx) => {
  let r = await agent(genPrompt(bp), { model: 'sonnet', phase: 'Generate', schema: WRITTEN, label: `gen:b${idx}` })
  if (r === null) {
    const singles = await parallel(Array.from({ length: BATCH }, (_, i) => () =>
      agent(singlePrompt(bp, i), { model: 'sonnet', phase: 'Generate', schema: WRITTEN, label: `gen1:b${idx}#${i}` })))
    r = { written: singles.filter(Boolean).flatMap((s) => s.written || []) }
  }
  return { written: r.written || [] }
})

const ok = results.filter(Boolean)
const total = ok.reduce((a, r) => a + r.written.length, 0)
log(`chunk done: ${total} hints written/confirmed across ${ok.length}/${count} batches`)
return { batches: ok.length, written: total }
