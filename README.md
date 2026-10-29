# PageRank on Three Networks

**CSE4039 Advanced Text & Social Media Analytics · Web Analytics project · VIT-AP University**
Swapnil (23BCB7031) · Shashikant (23BCB7077)

PageRank implemented from scratch (power iteration), verified against NetworkX, and applied to Zachary's Karate Club and a 100K-page Wikipedia link graph.

## Links

| | |
|---|---|
| 📊 Presentation | [17 Slides](https://claude.ai/artifact/JAcSMWMX8NUUN3Fnxu9Nvx) |
| 🖥️ Demo application | [PageRank Explorer (live)] · [`demo-app/index.html`](demo-app/index.html) |
| 🎬 Demo video | [Watch online (4:04)](https://claude.ai/artifact/JRoCjCB1EEv5RkgNCL2eSR) · [`video/PageRank_Demo_Video.mp4`](video/PageRank_Demo_Video.mp4) · [subtitles](video/PageRank_Demo_Video.srt) |
| 📄 Report (PDF) | [`docs/PageRank_Project_Report.pdf`](docs/PageRank_Project_Report.pdf) |
| 📓 Notebook | [`PageRank_Web_Analytics.ipynb`](PageRank_Web_Analytics.ipynb) |
| 🐍 Source code | [`src/pagerank_web_analytics.py`](src/pagerank_web_analytics.py) |
| 📥 Input data | [`data/`](data/) |
| 🔄 Intermediate data | [`intermediate/`](intermediate/) |
| 📈 Output data & charts | [`outputs/`](outputs/) |

## Results

| network | nodes | links | iterations | max \|ours − NetworkX\| |
|---|---:|---:|---:|---:|
| Karate Club | 34 | 78 | 45 | 1.5e-9 |
| Wikipedia 100K | 99,998 | 8,572,670 | 61 | 1.0e-7 |

![Karate Club network sized by PageRank](outputs/karate_network_pagerank.png)

## Run

```bash
pip install -r requirements.txt
jupyter notebook PageRank_Web_Analytics.ipynb   # Run All
```
Wikipedia data: see [`data/README.md`](data/README.md).
