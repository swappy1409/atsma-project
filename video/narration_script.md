# Demo video narration script

The video (`PageRank_Demo_Video.mp4`, 4:04, 1080p) shows this narration as burned-in subtitles. The same lines with timings are in `PageRank_Demo_Video.srt` (upload it as captions on YouTube, or read it aloud to record a voice-over in any video editor).

| time | narration |
|---|---|
| 00:00 | Demo: PageRank web analytics, implemented from scratch and tested on real networks. |
| 00:08 | PageRank models a random surfer: follow a link with probability d = 0.85, or jump to a random page. |
| 00:16 | pagerank_simple() translates the formula directly, using Python dictionaries. |
| 00:22 | Dangling pages (no out-links) would leak rank, so their mass is spread evenly over all pages. |
| 00:30 | Each iteration: PR(v) = (1 − d)/N + d · ( Σ PR(u)/L(u) + dangling/N ). |
| 00:35 | We stop when the total L1 change between two iterations drops below 10⁻⁸. |
| 00:42 | pagerank_sparse() does the same with a SciPy CSR matrix, so it scales to millions of links. |
| 00:49 | One sparse matrix–vector product per iteration replaces the per-node loops. |
| 00:56 | The demo app runs the same algorithm in the browser. First, a 4-page toy web. |
| 01:02 | Every page starts at 1/N = 0.25. Each click on Step is one power iteration. |
| 01:14 | After three steps C already leads, and D is falling toward its floor. |
| 01:25 | Run to convergence: 47 iterations. C ends highest because three pages link to it. |
| 01:32 | D has no in-links, so it keeps only the teleport share (1 − d)/N = 0.0375. |
| 01:39 | Dataset 1: Zachary's Karate Club, 34 members and 78 friendships. Colour shows the faction after the club split. |
| 01:49 | Animating the iterations: rank flows toward the best-connected members. |
| 01:56 | The two leaders, node 34 (administrator) and node 1 (instructor), get the highest PageRank. |
| 02:03 | Our values match NetworkX nx.pagerank to within 1.5 × 10⁻⁹. |
| 02:11 | Lower damping, d = 0.5: ranks move toward uniform and converge in only 19 iterations. |
| 02:21 | Higher damping, d = 0.95: ranks are more unequal (max/min 14×) but need 70 iterations. |
| 02:30 | Clicking a member highlights it. Node 3 bridges the two factions and ranks 4th. |
| 02:38 | Dataset 2: the Kaggle Wikipedia link graph, 99,998 pages and 8.57 million links after cleaning. |
| 02:46 | The sparse version converged in 61 iterations. The top 1% of pages hold 28.2% of all rank. |
| 02:53 | Citation identifiers such as ISBN, DOI and ISSN rank highest: almost every article cites them. |
| 03:00 | ISO has only 428 in-links yet ranks #16, because its links come from those top identifier pages. |
| 03:10 | Any edge list can be ranked. Here is a made-up 12-page university website. |
| 03:18 | Home ranks first because most pages link back to it: the navigation effect of site architecture. |
| 03:29 | Switching to undirected (social-network) links recomputes the ranking instantly. |
| 03:36 | The notebook saves every chart to outputs/. On a log scale convergence is a straight line: the error shrinks by a constant factor each step. |
| 03:47 | Damping sensitivity and the heavy-tailed Wikipedia distribution are explained in the report. |
| 03:55 | Source code, data, slides and the full report: github.com/swappy1409/atsma-project |
