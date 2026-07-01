import type { JobStatus } from "../api/client";

const steps = ["Uploading files", "Building RoleSpec", "Scoring candidates", "Generating explanations", "Completed"];

export default function ProcessingTimeline({ job }: { job: JobStatus | null }) {
  return (
    <section className="timeline-panel">
      <div className="progress-head">
        <span>{job?.stage || "Waiting"}</span>
        <strong>{job?.progress || 0}%</strong>
      </div>
      <div className="progress-track">
        <div style={{ width: `${job?.progress || 0}%` }} />
      </div>
      <p>{job?.message || "Upload files and run the pipeline to begin."}</p>
      <div className="timeline">
        {steps.map((step, index) => (
          <div key={step} className={(job?.progress || 0) >= index * 22 ? "timeline-step done" : "timeline-step"}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            {step}
          </div>
        ))}
      </div>
    </section>
  );
}
