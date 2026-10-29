# PageRank on three networks

**CSE4039 Advanced Text & Social Media Analytics · Web Analytics project · VIT-AP University**

We implement PageRank from scratch (power iteration), verify it against NetworkX's `nx.pagerank`, and apply it to Zachary's Karate Club and a 100,000-page Wikipedia link graph. A polite crawler for the VIT-AP website is included as a third network.

| | |
|---|---|
| Presentation | https://claude.ai/artifact/JAcSMWMX8NUUN3Fnxu9Nvx |
| Demo application | https://claude.ai/artifact/ENij4N89ywMKyusQ54Sq6j · or open [`demo-app/index.html`](demo-app/index.html) |
| Demo video | https://claude.ai/artifact/JRoCjCB1EEv5RkgNCL2eSR · or [`video/PageRank_Demo_Video.mp4`](video/PageRank_Demo_Video.mp4) |
| Full report (PDF) | [`docs/PageRank_Project_Report.pdf`](docs/PageRank_Project_Report.pdf) |

## Results at a glance

| network | type | nodes | links | iterations (tol 1e-8) | max \|ours − NetworkX\| |
|---|---|---:|---:|---:|---:|
| Karate Club | undirected social | 34 | 78 | 45 | 1.5e-9 |
| Wikipedia 100K | directed encyclopedia | 99,998 | 8,572,670 | 61 | 1.0e-7 (20K-page sample) |

* **Karate Club:** the two leaders who split the club, node 34 (administrator, PR 0.1009) and node 1 (instructor, PR 0.0970), rank first.
* **Wikipedia:** citation identifiers (ISBN, DOI, ISSN) dominate; the top 1% of pages hold 28.2% of all rank. *International Organization for Standardization* has only 428 in-links yet ranks 16th, because its links come from top-ranked pages.
* **Damping:** d = 0.5 converges in 19 iterations, d = 0.95 needs 70 and makes ranks far more unequal (max/min 4.7× → 14.2×); the top five never change.

## Repository structure
```
├── PageRank_Web_Analytics.ipynb   main notebook: theory, code, all networks, charts
├── src/pagerank_web_analytics.py  same code as a plain script (# %% cells)
├── requirements.txt
├── data/          input data (see data/README.md for the Wikipedia download)
├── intermediate/  transition matrices, iteration traces, adjacency list, damping sweep
├── outputs/       charts (PNG), top-k tables (CSV), results.json, summary.csv
├── demo-app/      PageRank Explorer, a single HTML file (no server needed)
├── video/         demo video (MP4), subtitles (SRT), narration script
├── docs/          project report (PDF) and links
└── pagerank-project.zip  original project bundle
```

## Run it (about 10 minutes)
```bash
pip install -r requirements.txt
# Wikipedia data: see data/README.md (or let kagglehub download it)
jupyter notebook PageRank_Web_Analytics.ipynb   # Run All
```
Karate Club needs no download. The VIT-AP crawl needs internet access (3–5 min, cached in `data/vitap_edges.csv`). Wikipedia (~28.9M raw links) needs 2–4 GB RAM; on a small laptop set `MAX_EDGES = 5_000_000`, or run on Colab/Kaggle.

## Core algorithm
```python
def pagerank_simple(graph, d=0.85, tol=1e-8, max_iter=200):
    # PR(v) = (1-d)/N + d * (sum_{u->v} PR(u)/L(u) + dangling_mass/N), until L1 change < tol
```
`pagerank_sparse()` does the same with a SciPy CSR matrix: one sparse matrix-vector product per iteration, which scales to millions of links.

## References
Brin & Page (1998), *The Anatomy of a Large-Scale Hypertextual Web Search Engine* · Zachary (1977), *An Information Flow Model for Conflict and Fission in Small Groups* · Rossi & Ahmed, *The Network Data Repository* · Kaggle: kutayahin/wikipedia-link-graph-100k · NetworkX.
