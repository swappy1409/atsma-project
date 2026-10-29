# Data

## `input/` (included)
| file | what it is |
|---|---|
| `karate_edges.csv` | Zachary's Karate Club, 78 undirected ties between 34 members (1-based labels as in the 1977 paper) |
| `toy_links.csv` | the 4-page toy web used in the worked example |
| `example_site_edges.csv` | a made-up 12-page university website, the sample input of the demo app's "Your own graph" view |

## Wikipedia Link Graph 100K (not included, ~28.9M links)
Download it from Kaggle: https://www.kaggle.com/datasets/kutayahin/wikipedia-link-graph-100k
and put the zip (zipped or unzipped) in `data/wikipedia/`. Or let the notebook fetch it with `kagglehub` (needs a Kaggle API token in `~/.kaggle/kaggle.json`; works out of the box on Kaggle and Colab).

## VIT-AP crawl (created on first run)
The notebook crawls https://vitap.ac.in/ (300 pages max, obeys robots.txt, 0.5 s between requests) and caches the edge list in `data/vitap_edges.csv`. Delete that file to crawl again.
