# Research Report: pgvector Indexing Strategies

pgvector supports two primary approximate nearest neighbor (ANN) index types — HNSW and IVFFlat — each with distinct performance characteristics, tunability, and operational trade-offs. Choosing the right index (and configuring it correctly) has significant implications for query latency, recall accuracy, memory consumption, and index build time, particularly at scale.

---

## Index Types Overview

pgvector provides two ANN index methods for accelerating vector similarity search: **HNSW** (Hierarchical Navigable Small World) and **IVFFlat** (Inverted File with Flat storage) [2][9][10]. Without an index, pgvector performs exact nearest-neighbor search via sequential scan, which guarantees perfect recall but does not scale [20].

Both index types support approximate search, meaning they trade some recall accuracy for significantly faster query performance [16]. The choice between them depends on the use case's priorities around build time, memory, query speed, and recall [4][17].

Beyond these two main types, pgvector also supports **half-precision indexing** and **binary quantization** as additional strategies for reducing memory footprint and improving throughput [21].

---

## HNSW: Architecture and Characteristics

HNSW builds a multi-layered graph structure where each node connects to its nearest neighbors. At query time, the search traverses from coarse upper layers to fine-grained lower layers, quickly narrowing in on candidates [7][23].

**Key creation syntax:**
```sql
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);
```
[3][11]

HNSW exposes two primary tuning parameters [3][7]:
- **`m`** — the number of bidirectional links per node (default: 16). Higher values improve recall but increase memory and build time.
- **`ef_construction`** — the size of the dynamic candidate list during index build (default: 64). Larger values improve index quality at the cost of build time.
- At query time, **`hnsw.ef_search`** controls the search beam width and can be adjusted per session to tune recall vs. latency.

HNSW's principal advantages are **superior query speed** and **high recall without requiring data to be loaded upfront** — you can insert rows and the index updates incrementally [4][13]. Benchmarks consistently show HNSW outperforming IVFFlat in most query latency scenarios [5][12].

The main drawbacks are **higher memory consumption** and **longer index build times**, since the full graph must be constructed and held in memory [4][6]. For very large datasets, the HNSW build process can be constrained by `maintenance_work_mem` settings, and strategies such as building the index in stages or increasing memory allocation are required [6].

---

## IVFFlat: Architecture and Characteristics

IVFFlat partitions the vector space into a configurable number of clusters (lists) using k-means, then stores vectors in their nearest cluster. At query time, only a subset of clusters is searched [7][15].

**Key creation syntax:**
```sql
CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
```
[9][22]

IVFFlat's primary tuning parameters are [7][15]:
- **`lists`** — the number of inverted lists (clusters). A common heuristic is `sqrt(n)` for up to 1M rows, or `rows / 1000` for larger datasets.
- **`ivfflat.probes`** — the number of clusters searched at query time (default: 1). Increasing probes improves recall at the cost of speed.

A critical constraint: **IVFFlat requires the table to be populated before the index is built**, because k-means clustering needs existing data to form meaningful partitions [4][17]. This makes it less suitable for tables with frequent inserts or rapidly changing data.

IVFFlat generally has **faster index build times** and **lower memory requirements** than HNSW [4][12], making it appealing for resource-constrained environments or large batch-indexed datasets. However, at equivalent recall levels, its query latency tends to be higher than HNSW [5][13].

---

## HNSW vs. IVFFlat: Comparative Trade-offs

| Dimension | HNSW | IVFFlat |
|---|---|---|
| Query speed | Faster [5][13] | Slower at equivalent recall [5] |
| Index build time | Slower [4][6] | Faster [4][12] |
| Memory usage | Higher [4][6] | Lower [4] |
| Incremental inserts | Supported [4] | Requires pre-populated data [4][17] |
| Recall ceiling | Generally higher [5][12] | Good, but probe-dependent [15] |

Benchmarks across dataset sizes show HNSW outperforming IVFFlat in most real-world query workloads [5][18]. For RAG (retrieval-augmented generation) applications specifically, AWS recommends evaluating both indexes against your specific dataset and recall requirements, as architecture differences in cluster probing vs. graph traversal produce varied results depending on data distribution [7].

One source notes that for very large datasets with tight memory budgets, IVFFlat may be preferable due to its lower memory overhead despite the query speed trade-off [6][4]. Sources do not disagree on the fundamental characteristics, but they do differ in emphasis: some highlight HNSW as the clear default choice [5][13], while others present IVFFlat as a viable and sometimes preferred option for specific operational constraints [4][17].

---

## Distance Metrics and Their Impact on Index Selection

pgvector supports multiple distance metrics, and the choice of metric must match the `ops` class used when creating the index [14][19]:

- **L2 distance** (`vector_l2_ops`) — Euclidean distance; suitable when absolute magnitude differences matter.
- **Cosine similarity** (`vector_cosine_ops`) — measures angular similarity; widely used for text embeddings where direction matters more than magnitude.
- **Inner product** (`vector_ip_ops`) — appropriate for normalized vectors or dot-product similarity models.
- **L1 distance** (`vector_l1_ops`) — Manhattan distance; useful for certain data distributions [19].

The choice of metric affects both accuracy and performance: using the wrong metric for a given embedding model can degrade recall significantly [14]. Both HNSW and IVFFlat support all distance metrics, but the index must be created with the correct operator class for the metric used at query time [9][22].

---

## Performance Optimization and Scaling Best Practices

Several practices are consistently recommended across sources for maximizing pgvector index performance:

**Memory configuration:** HNSW index builds are memory-intensive. Increasing `maintenance_work_mem` (e.g., to several gigabytes) before building the index can substantially reduce build time and improve graph quality [6][15]. Neon reports achieving up to 30× faster HNSW index builds through memory and configuration tuning [25].

**Build timing:** For IVFFlat, always build the index after the bulk of data is loaded, not before [4][17]. For HNSW, incremental builds are possible but large datasets may benefit from bulk loading followed by index creation.

**Quantization:** For very large datasets, half-precision indexing and binary quantization can reduce memory usage while preserving acceptable recall, enabling HNSW to scale further [6][21].

**Session-level tuning:** Adjusting `hnsw.ef_search` or `ivfflat.probes` at the session level allows fine-grained control over the recall-latency trade-off without rebuilding indexes [15][26].

**Schema and query design:** Ensuring queries use the same operator class as the index (e.g., `<=>` for cosine, `<->` for L2) is required for the index to be used at all [24][9]. Partial indexes (e.g., filtering by a tenant ID) can further improve performance in multi-tenant scenarios [26].

**Benchmarking at scale:** Performance characteristics shift as dataset size grows. Sources recommend benchmarking both index types against representative data volumes, as recall and latency curves diverge non-linearly across small, medium, and large corpora [18][8].

---

## Sources

[1] Index Performance and Comparison | pgvector/pgvector | DeepWiki — https://deepwiki.com/pgvector/pgvector/5.3-index-performance-and-comparison (web)

[2] What is pgvector? | Databricks — https://www.databricks.com/blog/what-is-pgvector (web)

[3] HNSW indexes | Supabase Docs — https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes (web)

[4] IVFFlat vs HNSW in pgvector: Which Index Should You Use? — https://dev.to/philip_mcclarence_2ef9475/ivfflat-vs-hnsw-in-pgvector-which-index-should-you-use-305p (web)

[5] pgvector: HNSW vs IVFFlat Benchmarks in Postgres (2026) — https://pecollective.com/tools/pgvector/ (web)

[6] Scaling pgvector: Memory, Quantization, and Index Build Strategies — https://mydba.dev/blog/pgvector-scaling-large-datasets (web)

[7] Optimize generative AI applications with pgvector indexing: A deep dive into IVFFlat and HNSW techniques — https://aws.amazon.com/blogs/database/optimize-generative-ai-applications-with-pgvector-indexing-a-deep-dive-into-ivfflat-and-hnsw-techniques/ (web)

[8] Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes? — http://arxiv.org/abs/2409.06464v1 (arxiv)

[9] Vector Similarity Search with PostgreSQL's pgvector - A Deep Dive | Severalnines — https://severalnines.com/blog/vector-similarity-search-with-postgresqls-pgvector-a-deep-dive/ (web)

[10] How to Use pgvector for Similarity Search on Heroku Postgres | Heroku — https://blog.heroku.com/pgvector-for-similarity-search-on-heroku-postgres (web)

[11] Polymath Engineer Weekly #104 - by Felipe Alcantara — https://weekly.polymathengineer.dev/p/104 (web)

[12] PGVector: HNSW vs IVFFlat — A Comprehensive Study — https://medium.com/@bavalpreetsinghh/pgvector-hnsw-vs-ivfflat-a-comprehensive-study-21ce0aaab931 (web)

[13] PgVector indexing options for vector similarity search — https://omiid.me/notebook/31/pgvector-indexing-options-for-vector-similarity-search (web)

[14] Distance Metrics Overview | pgvector/pgvector | DeepWiki — https://deepwiki.com/pgvector/pgvector/4.1-distance-metrics-overview (web)

[15] Performance Tips Using Postgres and pgvector - Crunchy Data — https://www.crunchydata.com/blog/pgvector-performance-for-developers (web)

[16] Faster similarity search performance with pgvector indexes | Google Cloud Blog — https://cloud.google.com/blog/products/databases/faster-similarity-search-performance-with-pgvector-indexes (web)

[17] 05 - pgvector Indexing Strategies: HNSW vs IVFFlat — https://www.federicocalo.dev/en/blog/pgvector-indexing-strategies-hnsw-vs-ivfflat (web)

[18] Benchmarking pgvector RAG performance across different dataset sizes — https://mastra.ai/blog/pgvector-perf (web)

[19] Speed up PostgreSQL® pgvector queries with indexes - Aiven — https://aiven.io/developer/postgresql-pgvector-indexes (web)

[20] Faster similarity search performance with pgvector indexes | Google Cloud — https://cloud.google.com/blog/products/databases/faster-similarity-search-performance-with-pgvector-indexes/ (web)

[21] GitHub - pgvector/pgvector: Open-source vector similarity search for Postgres — https://github.com/pgvector/pgvector (web)

[22] Creating Your First Index | pgvector/pgvector | DeepWiki — https://deepwiki.com/pgvector/pgvector/2.3-creating-your-first-index (web)

[23] Understanding pgvector's HNSW Index Storage in Postgres — https://lantern.dev/blog/pgvector-storage (web)

[24] PostgreSQL pgvector and RAG: Best Practices and Examples for Better Results — https://postgresqlhtx.com/postgresql-pgvector-and-rag-best-practices-and-examples-for-better-results/ (web)

[25] pgvector: 30x Faster Index Build for your Vector Embeddings — https://neon.com/blog/pgvector-30x-faster-index-build-for-your-vector-embeddings (web)

[26] Make Postgres Feel Instant with These 10 pgvector Indexing Tricks — https://medium.com/@sparknp1/make-postgres-feel-instant-with-these-10-pgvector-indexing-tricks-69247d938713 (web)