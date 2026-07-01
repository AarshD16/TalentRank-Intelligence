import { useEffect, useState } from "react";
import { createJob, getJob, getResults, type CandidateResult, type JobResults, type JobStatus } from "../api/client";
import UploadPanel, { type UploadState } from "../components/UploadPanel";
import ProcessingTimeline from "../components/ProcessingTimeline";
import MetricsCards from "../components/MetricsCards";
import CandidateTable from "../components/CandidateTable";
import CandidateDrawer from "../components/CandidateDrawer";
import RoleSpecPanel from "../components/RoleSpecPanel";
import ExportPanel from "../components/ExportPanel";
import EmptyState from "../components/EmptyState";

const initialUpload: UploadState = {
  jobDescription: null,
  candidates: null,
  shortlistSize: 100,
  debug: true
};

export default function Dashboard() {
  const [upload, setUpload] = useState(initialUpload);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [results, setResults] = useState<JobResults | null>(null);
  const [selected, setSelected] = useState<CandidateResult | null>(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("overview");

  const running = job?.status === "queued" || job?.status === "running";

  async function run() {
    setError("");
    setResults(null);
    const form = new FormData();
    if (upload.jobDescription) form.append("job_description", upload.jobDescription);
    if (upload.candidates) form.append("candidates", upload.candidates);
    form.append("rolespec_mode", "auto_llm");
    form.append("shortlist_size", String(upload.shortlistSize));
    form.append("debug", String(upload.debug));
    try {
      const created = await createJob(form);
      setJobId(created.job_id);
      setJob({ job_id: created.job_id, status: "queued", progress: 0, stage: "Queued", message: "Job queued", metrics: {} });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create job");
    }
  }

  useEffect(() => {
    if (!jobId) return;
    const timer = window.setInterval(async () => {
      try {
        const status = await getJob(jobId);
        setJob(status);
        if (status.status === "completed") {
          window.clearInterval(timer);
          const payload = await getResults(jobId);
          setResults(payload);
        }
        if (status.status === "failed") {
          window.clearInterval(timer);
          setError(status.message);
        }
      } catch (err) {
        window.clearInterval(timer);
        setError(err instanceof Error ? err.message : "Polling failed");
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [jobId]);

  return (
    <main>
      <section className="hero">
        <div>
          <span className="eyebrow">TalentRank Intelligence</span>
          <h1>Evidence-first candidate ranking for serious recruiting teams.</h1>
          <p>Compile a JD into a RoleSpec, score career evidence deterministically, and export a shortlist recruiters can defend.</p>
        </div>
        <div className="hero-panel">
          <span>CLI preserved</span>
          <strong>FastAPI + React</strong>
          <small>Same ranking engine, premium browser workflow.</small>
        </div>
      </section>

      <UploadPanel state={upload} setState={setUpload} onRun={run} running={running} />
      <ProcessingTimeline job={job} />
      {error && <div className="error-banner">{error}</div>}

      {results ? (
        <section className="results">
          <div className="tabs">
            {["overview", "top100", "rolespec", "export"].map((tab) => (
              <button key={tab} className={activeTab === tab ? "active" : ""} onClick={() => setActiveTab(tab)}>
                {tab === "top100" ? "Top 100" : tab}
              </button>
            ))}
          </div>
          {activeTab === "overview" && <MetricsCards metrics={results.metrics} />}
          {activeTab === "top100" && <CandidateTable candidates={results.top_candidates} onSelect={setSelected} />}
          {activeTab === "rolespec" && <RoleSpecPanel rolespec={results.rolespec} />}
          {activeTab === "export" && <ExportPanel jobId={jobId} />}
        </section>
      ) : (
        <EmptyState />
      )}
      <CandidateDrawer candidate={selected} onClose={() => setSelected(null)} />
    </main>
  );
}
