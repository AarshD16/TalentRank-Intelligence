import type { CandidateResult } from "../api/client";

type Props = {
  candidate: CandidateResult | null;
  onClose: () => void;
};

export default function CandidateDrawer({ candidate, onClose }: Props) {
  if (!candidate) return null;
  const components = candidate.components || {};
  return (
    <aside className="drawer-backdrop" onClick={onClose}>
      <div className="drawer" onClick={(event) => event.stopPropagation()}>
        <button className="drawer-close" onClick={onClose}>Close</button>
        <span className="eyebrow">Rank {candidate.rank}</span>
        <h2>{candidate.title || candidate.candidate_id}</h2>
        <strong className="drawer-score">{candidate.score.toFixed(4)}</strong>
        <p>{candidate.reasoning}</p>

        <h3>Score breakdown</h3>
        <div className="bars">
          {Object.entries(components).slice(0, 12).map(([name, value]) => (
            <div key={name}>
              <span>{name.replace("competency_", "").replaceAll("_", " ")}</span>
              <div><i style={{ width: `${Math.min(100, Number(value) * 100)}%` }} /></div>
            </div>
          ))}
        </div>

        <h3>Flags and evidence</h3>
        <ul>
          {(candidate.score_flags || "No major flags").split(" | ").map((flag) => <li key={flag}>{flag}</li>)}
        </ul>
        {candidate.top_competencies && <p className="drawer-muted">{candidate.top_competencies}</p>}
      </div>
    </aside>
  );
}
