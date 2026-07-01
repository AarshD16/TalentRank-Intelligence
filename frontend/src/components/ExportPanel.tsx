import { getDownloadUrl } from "../api/client";

export default function ExportPanel({ jobId }: { jobId: string | null }) {
  return (
    <section className="export-panel">
      <h2>Exports</h2>
      <p>Download the same CSV contract produced by the CLI.</p>
      <div className="export-actions">
        <a className={!jobId ? "disabled" : ""} href={jobId ? getDownloadUrl(jobId, "submission") : "#"}>Download submission.csv</a>
        <a className={!jobId ? "disabled" : ""} href={jobId ? getDownloadUrl(jobId, "debug") : "#"}>Download debug CSV</a>
      </div>
      <code>python rank.py --candidates candidates.jsonl --role configs/rolespec_redrob_senior_ai_engineer.yaml --out submission.csv</code>
    </section>
  );
}
