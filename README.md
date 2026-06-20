# redrob-hackathon

Built for the Redrob Intelligent Candidate Discovery Challenge.

The problem: given 100,000 candidate profiles and a job description, find and rank the best 100 people for the role. Most teams will run embeddings and call it done. I didn't.

The insight that drove my approach is that a great-looking resume from someone who hasn't logged in for eight months and replies to 5% of recruiter messages is, for practical hiring purposes, not actually available. So I treat availability and responsiveness as first-class signals, not afterthoughts.

## How the pipeline works

The system runs in three passes.

The first pass is pure logic. It reads all 100,000 profiles one by one without loading them into memory and immediately discard anyone who can't possibly fit: wrong location and unwilling to relocate, experience too far outside the JD range, non-engineering job title or a profile with logical impossibilities that signal a honeypot (claiming ten years of experience in a technology that didn't exist ten years ago, for example).

The second pass scores what's left. Each surviving candidate gets a composite score from three sources: how semantically similar their profile is to the JD using a sentence-transformer model, how well their specific skills match the terms and emphasis in the JD text and a behavioral multiplier derived from their platform activity — last login date, recruiter response rate, notice period, GitHub activity, interview completion rate. The behavioral signal is applied as a multiplier, so a ghost candidate with a perfect resume still gets crushed.

The third pass writes the output. Top 100 by composite score, with a reasoning string for each that references actual facts from their profile — no templates, no hallucination.

## Running it

```bash
pip install sentence-transformers scikit-learn numpy scipy
python run_pipeline.py --candidates candidates.jsonl --jd job_description.txt --out team_apex.csv
```

To test on the sample file:
```bash
python run_pipeline.py --candidates sample_candidates.json --jd job_description.txt --sample-mode
```

All flags except --candidates have sensible defaults (--jd defaults to job_description.txt, --out defaults to team_apex.csv), so the pipeline runs end-to-end even with minimal arguments.

The pipeline works with any job description. Point `--jd` at any plain text file and it will parse the experience range, locations and relevant skills automatically — nothing about the role is hardcoded.

## What's in this repo

`README.md` — this file  
`run_pipeline.py` — the full pipeline, single file, no hidden steps  
`job_description.txt` — the JD we ranked against  
`validate_submission.py` — format checker from the hackathon bundle  
`requirements.txt` — dependencies  
`team_apex.csv` — my final submission