import { useMemo, useState } from "react";
import type { CandidateResult } from "../api/client";

type Props = {
  candidates: CandidateResult[];
  onSelect: (candidate: CandidateResult) => void;
};

export default function CandidateTable({ candidates, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    return candidates.filter((candidate) =>
      [candidate.candidate_id, candidate.title, candidate.reasoning].join(" ").toLowerCase().includes(q)
    );
  }, [candidates, query]);

  return (
    <section className="table-panel">
      <div className="table-toolbar">
        <h2>Top 100</h2>
        <input placeholder="Search candidate, title, evidence..." value={query} onChange={(event) => setQuery(event.target.value)} />
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Candidate</th>
              <th>Score</th>
              <th>Title</th>
              <th>Years</th>
              <th>Reasoning</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((candidate) => (
              <tr key={candidate.candidate_id} onClick={() => onSelect(candidate)}>
                <td>{candidate.rank}</td>
                <td>{candidate.candidate_id}</td>
                <td>{candidate.score.toFixed(4)}</td>
                <td>{candidate.title || "-"}</td>
                <td>{candidate.years || "-"}</td>
                <td>{candidate.reasoning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
