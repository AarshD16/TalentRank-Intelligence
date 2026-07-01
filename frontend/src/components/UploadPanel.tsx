import type { ChangeEvent } from "react";

export type UploadState = {
  jobDescription: File | null;
  candidates: File | null;
  shortlistSize: number;
  debug: boolean;
};

type Props = {
  state: UploadState;
  setState: (state: UploadState) => void;
  onRun: () => void;
  running: boolean;
};

export default function UploadPanel({ state, setState, onRun, running }: Props) {
  const canRun = Boolean(state.candidates && state.jobDescription);

  const setFile = (key: "jobDescription" | "candidates") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      setState({ ...state, [key]: event.target.files?.[0] || null });
    };

  return (
    <section className="workflow-panel">
      <div className="step-card">
        <span className="step-index">01</span>
        <h3>Upload Job Description</h3>
        <p>Use .docx, .txt, or .md when compiling a fresh policy.</p>
        <label className="file-box">
          <input type="file" accept=".docx,.txt,.md" onChange={setFile("jobDescription")} />
          <strong>{state.jobDescription?.name || "Choose JD file"}</strong>
          <small>{fileMeta(state.jobDescription) || "Required for smart RoleSpec selection"}</small>
        </label>
      </div>

      <div className="step-card">
        <span className="step-index">02</span>
        <h3>Upload Candidates</h3>
        <p>Upload candidates.jsonl or candidates.json from the Redrob schema.</p>
        <label className="file-box required">
          <input type="file" accept=".jsonl,.json" onChange={setFile("candidates")} />
          <strong>{state.candidates?.name || "Choose candidates file"}</strong>
          <small>{fileMeta(state.candidates) || "Required"}</small>
        </label>
      </div>

      <div className="step-card wide">
        <span className="step-index">03</span>
        <h3>Smart RoleSpec Policy</h3>
        <div className="auto-policy-card">
          <strong>Automatic JD intelligence</strong>
          <p>
            TalentRank checks whether this JD already has a cached LLM RoleSpec. If it does, ranking starts immediately.
            If not, the backend compiles a fresh scoring policy with the configured LLM and saves it for future runs.
          </p>
          <div className="policy-list">
            <span>Cache-first reuse</span>
            <span>LLM compile on miss</span>
            <span>Deterministic ranking after policy creation</span>
          </div>
        </div>
        <div className="config-row">
          <label>
            Shortlist size
            <input
              type="number"
              min={1}
              max={100}
              value={state.shortlistSize}
              onChange={(event) => setState({ ...state, shortlistSize: Number(event.target.value) })}
            />
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={state.debug}
              onChange={(event) => setState({ ...state, debug: event.target.checked })}
            />
            Keep extended debug pool
          </label>
        </div>
      </div>

      <div className="run-card">
        <span className="step-index">04</span>
        <h3>Run Ranking</h3>
        <p>Launch deterministic scoring and export generation.</p>
        <button className="run-button" disabled={!canRun || running} onClick={onRun}>
          {running ? "TalentRank is running" : "Run TalentRank"}
        </button>
      </div>
    </section>
  );
}

function fileMeta(file: File | null) {
  if (!file) return "";
  return `${(file.size / 1024 / 1024).toFixed(2)} MB`;
}
