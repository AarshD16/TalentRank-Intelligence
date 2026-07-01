export default function MetricsCards({ metrics }: { metrics: Record<string, unknown> }) {
  const cards = [
    ["Processed", String(metrics.candidates_processed || 0), "Candidate records"],
    ["Top score", number(metrics.top_score), "Highest rank score"],
    ["Complete match", String(metrics.complete_match_count || 0), "Bundle matches"],
    ["Evaluation", String(metrics.evaluation_evidence_count || 0), "Evidence count"],
    ["Vector", String(metrics.vector_evidence_count || 0), "Hybrid evidence"],
    ["Runtime", `${number(metrics.runtime_seconds)}s`, "Backend time"]
  ];
  return (
    <section className="metrics-grid">
      {cards.map(([label, value, caption]) => (
        <div className="metric" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
          <small>{caption}</small>
        </div>
      ))}
    </section>
  );
}

function number(value: unknown) {
  const n = Number(value || 0);
  return Number.isFinite(n) ? n.toFixed(n > 10 ? 0 : 3) : "0";
}
