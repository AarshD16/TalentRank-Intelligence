export type JobStatus = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  stage: string;
  message: string;
  metrics: Record<string, unknown>;
};

export type CandidateResult = {
  candidate_id: string;
  rank: number;
  score: number;
  title?: string;
  years?: string;
  reasoning: string;
  score_flags?: string;
  missing_must_have?: string;
  top_competencies?: string;
  components?: Record<string, number>;
};

export type JobResults = {
  top_candidates: CandidateResult[];
  metrics: Record<string, unknown>;
  rolespec: any;
  diagnostics: Record<string, unknown>;
};

export async function createJob(formData: FormData): Promise<{ job_id: string; status: string }> {
  const response = await fetch("/api/jobs", { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function getResults(jobId: string): Promise<JobResults> {
  const response = await fetch(`/api/jobs/${jobId}/results`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export function getDownloadUrl(jobId: string, type: "submission" | "debug"): string {
  return `/api/jobs/${jobId}/download/${type}`;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}
