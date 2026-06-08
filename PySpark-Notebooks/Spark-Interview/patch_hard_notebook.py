"""
patch_hard_notebook.py
Run once:  python patch_hard_notebook.py
Appends all missing cells (Topics 4-8 gaps + final spark.stop()) to
03-Hard-Spark-Interview-Prep.ipynb without touching existing cells.
"""
import json, uuid, copy, os

NB_PATH = os.path.join(os.path.dirname(__file__), "03-Hard-Spark-Interview-Prep.ipynb")

def cid():
    return uuid.uuid4().hex[:8]

def md(src):
    return {"cell_type": "markdown", "id": cid(), "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "id": cid(), "metadata": {}, "execution_count": None, "outputs": [], "source": src}

# ── helpers ──────────────────────────────────────────────────────────────────

NEW_CELLS = []  # will be appended in order after the existing last cell

# ════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — Executor Memory: missing cells I (exercise), J (benchmark), L (followup)
# These go BEFORE the existing t5 cells.  We'll insert by finding f605b372 later.
# ════════════════════════════════════════════════════════════════════════════

T4_EXERCISE_ID = "t4_exercise"
T4_BENCH_ID    = "t4_bench"
T4_FOLLOWUP_ID = "t4_followup"

t4_exercise = code("""\
# ── Topic 4 · Cell I: Optimization Exercise ──────────────────────────────
# Fix each TODO then compare Spark UI > Stages (spill bytes should drop to 0).
from pyspark.sql import functions as F
from pyspark.sql.window import Window

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)

# TODO 1: Change Window.unboundedPreceding → -6 to bound memory per partition.
win = Window.partitionBy("Company").orderBy("Date").rowsBetween(
    Window.unboundedPreceding, Window.currentRow   # ← fix this
)

# TODO 2: Uncomment and set correct values (target: more execution, less storage).
# spark.conf.set("spark.memory.fraction", "???")        # default 0.6 → try 0.75
# spark.conf.set("spark.memory.storageFraction", "???") # default 0.5 → try 0.3

# TODO 3: Move .cache() to AFTER the filter so only filtered rows occupy storage pool.
enriched = cabs.withColumn("running_trips", F.count("*").over(win)).cache()  # ← move cache
filtered  = enriched.filter(F.col("Fare") > 10)

# TODO 4: Replace groupBy+agg with rdd.aggregateByKey for map-side pre-reduction.
result = filtered.groupBy("Company").agg(F.sum("Fare").alias("total_fare"))

result.orderBy(F.col("total_fare").desc()).show(5)
print("Check Spark UI Stages: 'Shuffle Spill (memory)' and '(disk)' should be 0.")\
""")
t4_exercise["id"] = T4_EXERCISE_ID

t4_bench = code("""\
# ── Topic 4 · Cell J: Performance Benchmarking ───────────────────────────
import time
from pyspark.sql import functions as F
from pyspark.sql.window import Window

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)

# ── BEFORE: unbounded window, default memory config ──
spark.conf.set("spark.memory.fraction", "0.6")
spark.conf.set("spark.memory.storageFraction", "0.5")

win_bad = Window.partitionBy("Company").orderBy("Date").rowsBetween(
    Window.unboundedPreceding, Window.currentRow
)
t0 = time.time()
(cabs.withColumn("r", F.count("*").over(win_bad))
     .filter(F.col("Fare") > 10)
     .groupBy("Company").agg(F.sum("Fare").alias("total"))
     .count())
bad_time = time.time() - t0
print(f"BEFORE (unbounded window, default memory): {bad_time:.2f}s")

# ── AFTER: bounded window, tuned memory config ──
spark.conf.set("spark.memory.fraction", "0.75")
spark.conf.set("spark.memory.storageFraction", "0.3")

win_good = Window.partitionBy("Company").orderBy("Date").rowsBetween(-6, 0)
t0 = time.time()
(cabs.withColumn("r", F.count("*").over(win_good))
     .filter(F.col("Fare") > 10)
     .groupBy("Company").agg(F.sum("Fare").alias("total"))
     .count())
good_time = time.time() - t0
print(f"AFTER  (bounded window, tuned memory):      {good_time:.2f}s")
print(f"Speedup: {bad_time/good_time:.1f}x  (more dramatic at 100M+ rows in production)")\
""")
t4_bench["id"] = T4_BENCH_ID

t4_followup = md("""\
### Topic 4 — Common Follow-up Questions

1. **"spark.memory.fraction is 0.6 and storageFraction is 0.5. How many bytes of execution memory does a 4 GB executor actually get?"**
   Walk through: 4 GB × 0.6 = 2.4 GB unified; × (1 − 0.5) = 1.2 GB execution at steady state.  Storage can borrow from execution and vice versa, but execution evicts storage under pressure.

2. **"Your WindowExec spills to disk even though the executor has 8 GB. Why?"**
   WindowExec buffers one partition's full sort-order frame.  If a single partition holds 500 M rows (data skew + unpartitioned window), it will exceed the execution pool regardless of total executor size.  Fix: add a meaningful `partitionBy` clause.

3. **"What is `spark.executor.memoryOverhead` and when would you increase it?"**
   Off-JVM-heap memory reserved for Python workers (PySpark UDFs), native memory allocations, and JVM internals (thread stacks, NIO buffers).  Default: `max(384 MB, 0.10 × executorMemory)`.  Increase when you see container-killed OOM in YARN/Kubernetes logs *outside* the JVM heap, e.g., large Arrow batches in pandas UDFs.

4. **"BytesToBytesMap ran out of space and fell back to sort-based aggregation. What does that mean in terms of Spark UI?"**
   Hash aggregate (HashAggregateExec) switches to SortAggregateExec mid-task when `BytesToBytesMap` exceeds `spark.sql.codegen.aggregate.map.twolevel.enabled` thresholds.  You'll see an extra `Sort` node appear in the physical plan and task memory spill metrics increase.

5. **"How does `spark.memory.offHeap.enabled` interact with `spark.memory.fraction`?"**
   Off-heap memory is *outside* the fraction model — it is a fixed pool (`spark.memory.offHeap.size`) managed separately.  Enabling it lets UnsafeRow store data in native memory, reducing GC pressure but requiring `sun.misc.Unsafe` access.  AQE cannot shuffle off-heap data directly; serialisation still happens on-heap at the exchange boundary.\
""")
t4_followup["id"] = T4_FOLLOWUP_ID

# ════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — Shuffle Spill: missing cells I (exercise), K (best practices), L (followup)
# ════════════════════════════════════════════════════════════════════════════

T5_EXERCISE_ID  = "t5_exercise"
T5_BEST_ID      = "t5_best"
T5_FOLLOWUP_ID  = "t5_followup"

t5_exercise = code("""\
# ── Topic 5 · Cell I: Optimization Exercise ──────────────────────────────
# Each TODO targets a specific shuffle-spill root cause.
import time
from pyspark.sql import functions as F

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)

# TODO 1: shuffle.partitions=200 creates 200 tasks for a tiny dataset.
#         Set advisoryPartitionSizeInBytes=64m so AQE coalesces them.
# spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "???")  # try "64m"

# TODO 2: Kryo serialiser cuts shuffle bytes ~30%.  Add two configs:
# spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
# spark.conf.set("spark.kryo.registrationRequired", "false")

# TODO 3: reduce.maxSizeInFlight=48m limits how much each reducer fetches per round.
#         For large shuffles, increase to 96m to reduce fetch round-trips.
# spark.conf.set("spark.reducer.maxSizeInFlight", "???")  # try "96m"

# TODO 4: shuffle.file.buffer=32k means many small writes to the shuffle file.
#         Increase to 1m to batch writes (especially helpful on HDDs / network storage).
# spark.conf.set("spark.shuffle.file.buffer", "???")  # try "1m"

t0 = time.time()
result = (cabs
    .groupBy("Company", "Payment_Type")
    .agg(F.sum("Fare").alias("total"), F.avg("Tip").alias("avg_tip"))
    .orderBy(F.col("total").desc()))
result.show(5)
print(f"Elapsed: {time.time()-t0:.2f}s — check Spark UI Stages for spill bytes")\
""")
t5_exercise["id"] = T5_EXERCISE_ID

t5_best = md("""\
### Topic 5 — Production Best Practices: Shuffle Spill

1. **Right-size shuffle partitions with AQE** — set `spark.sql.adaptive.advisoryPartitionSizeInBytes=64m` instead of hard-coding `spark.sql.shuffle.partitions`.  AQE will coalesce or split at runtime.

2. **Enable Kryo serialisation** — `spark.serializer=org.apache.spark.serializer.KryoSerializer` reduces shuffle payload by 20–40% for complex types; register domain classes with `spark.kryo.classesToRegister`.

3. **Tune the write buffer** — `spark.shuffle.file.buffer` (default 32 KB) controls the in-memory write buffer per shuffle output file.  Set to 1 MB on cloud object storage or network-attached volumes where small-write amplification is expensive.

4. **Tune the fetch buffer** — `spark.reducer.maxSizeInFlight` (default 48 MB) caps how much shuffle data a reducer requests per round-trip.  Increase to 96–128 MB on high-bandwidth clusters to reduce fetch latency.

5. **Compress shuffle data** — `spark.shuffle.compress=true` (default) with `spark.io.compression.codec=lz4` gives the best balance of CPU cost vs bytes on wire.  Use `zstd` for CPU-constrained clusters where IO is the bottleneck.

6. **Use External Shuffle Service** — on YARN/Kubernetes, enable `spark.shuffle.service.enabled=true` so shuffle files survive executor eviction and avoid re-computation on speculation.

7. **Partition before heavy aggregations** — `repartition("join_key")` before a large groupBy co-locates data, turning a shuffle-heavy aggregate into a local aggregate.  Measure with `df.explain()` — look for `Exchange` nodes.

8. **Monitor with Spark UI** — Stages tab: *Shuffle Write* (bytes written by mappers) and *Shuffle Read* (bytes fetched by reducers).  *Spill (memory)* and *Spill (disk)* indicate the executor ran out of execution memory during the sort phase.

9. **Bucket frequently-joined tables** — bucketing on the join key eliminates the shuffle exchange entirely, replacing it with a `BucketedScan`.  Requires both tables bucketed on the same key with the same bucket count.

10. **Prefer reduceByKey over groupByKey on RDDs** — `groupByKey` ships all values to the reducer before aggregating; `reduceByKey` applies the combiner on the map side, cutting shuffle bytes proportionally to the reduction ratio.\
""")
t5_best["id"] = T5_BEST_ID

t5_followup = md("""\
### Topic 5 — Common Follow-up Questions

1. **"What is the difference between Shuffle Spill (memory) and Shuffle Spill (disk) in the Spark UI?"**
   *Spill (memory)*: bytes de-serialised from in-memory records before they were written to disk (the "logical" size of spilled data).  *Spill (disk)*: bytes actually written to disk (compressed + serialised, typically 3–5× smaller).  A large memory/disk ratio suggests inefficient serialisation.

2. **"SortShuffleManager writes one file per reducer per mapper in 'bypass' mode and one sorted file per mapper in normal mode. When does it use each?"**
   Bypass (hash shuffle) mode activates when the number of output partitions ≤ `spark.shuffle.sort.bypassMergeThreshold` (default 200) AND there is no map-side aggregation.  Normal (sort) mode is used otherwise — it sorts records by partition ID before writing a single sorted file + index file per map task.

3. **"You increased spark.sql.shuffle.partitions from 200 to 2000 and spill went away, but the job got slower. Why?"**
   More partitions → more tasks → higher task-scheduling overhead and more small shuffle files (small-file problem for the reducer).  AQE's `coalescePartitions` should handle this automatically; if it doesn't, verify `spark.sql.adaptive.enabled=true` and check `isFinalPlan` in `.explain("formatted")`.

4. **"How does tungsten sort differ from Java sort in the context of shuffle spill?"**
   Tungsten's `UnsafeExternalSorter` operates on `UnsafeRow` pointers (8-byte records: partition ID + row pointer), sorting pointers rather than full objects.  This reduces the memory footprint of the sort buffer and avoids GC pressure.  Spill files are written as serialised `UnsafeRow` byte arrays, not Java-serialised objects.

5. **"When would you use sort-based shuffle vs hash-based shuffle?"**
   Hash shuffle (bypass mode) is faster for low partition counts because it avoids the sort phase.  Sort shuffle is mandatory for high partition counts (> 200) and for any operation requiring sorted output (SortMergeJoin, ordered aggregates).  In practice, AQE's `coalescePartitions` + a modest `shuffle.partitions` value makes sort shuffle the default safe choice.\
""")
t5_followup["id"] = T5_FOLLOWUP_ID

# ════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Dynamic Partitioning: missing cells F, G, H, I, J, L, M
# ════════════════════════════════════════════════════════════════════════════

T6_LOGICAL_ID  = "t6_logical"
T6_PHYSICAL_ID = "t6_physical"
T6_UI_ID       = "t6_ui"
T6_EXERCISE_ID = "t6_exercise"
T6_BENCH_ID    = "t6_bench"
T6_FOLLOWUP_ID = "t6_followup"
T6_SOLUTION_ID = "t6_solution"

t6_logical = md("""\
### Topic 6 · Cell F: Expected Logical Plan Discussion

**Query**: `df.write.partitionBy("payment_type").mode("overwrite").parquet(path)`

```
== Analyzed Logical Plan ==
InsertIntoHadoopFsRelationCommand(path, partitionBy=[payment_type],
  overwrite=true, staticPartitions={})
  Repartition(200, shuffle=true)    ← added by Spark to distribute by partition key
    Relation[...] ParquetRelation

== Optimized Logical Plan ==
InsertIntoHadoopFsRelationCommand
  RepartitionByExpression [payment_type]   ← Catalyst rewrites to hash-partition by key
    Filter isnotnull(payment_type)         ← pushed down to avoid null partition dirs
    Relation[...] ParquetRelation
```

**Key optimizer rules applied**:
- `ReplaceExceptWithAntiJoin` — not relevant here, but partition overwrite mode affects which rule handles the write
- `PushPredicateThroughJoin` — null filter on partition key pushed below the repartition
- With `STATIC` overwrite mode: Catalyst wraps the write in `OverwriteByExpression` which drops ALL existing partitions before writing
- With `DYNAMIC` overwrite mode (`spark.sql.sources.partitionOverwriteMode=DYNAMIC`): only partitions present in the new data are replaced — Catalyst inserts a `PartitionFiltering` node\
""")
t6_logical["id"] = T6_LOGICAL_ID

t6_physical = md("""\
### Topic 6 · Cell G: Expected Physical Plan Discussion

```
== Physical Plan ==
Execute InsertIntoHadoopFsRelationCommand
+- *(2) Sort [payment_type ASC NULLS FIRST], false, 0
   +- Exchange hashpartitioning(payment_type, 200), ENSURE_REQUIREMENTS
      +- *(1) Filter isnotnull(payment_type)
         +- *(1) FileScan parquet [trip_id, fare, payment_type]
                 Batched: true, DataFilters: [isnotnull(payment_type)]
                 PartitionFilters: []
                 PushedFilters: [IsNotNull(payment_type)]
                 ReadSchema: struct<...>
```

**Node-by-node explanation**:

| Node | What it does | Watch in Spark UI |
|------|-------------|-------------------|
| `FileScan` | Reads parquet with column pruning | *numFiles*, *bytes read* |
| `Filter isnotnull` | Drops null partition keys (avoids `__HIVE_DEFAULT_PARTITION__`) | Task count in Stage 1 |
| `Exchange hashpartitioning` | Shuffles rows so all rows for one partition land on same task | Shuffle Write bytes |
| `Sort [payment_type]` | Within each task, sorts by partition key so writer outputs one file per partition key contiguously | Sort spill metrics |
| `InsertIntoHadoopFsRelationCommand` | Calls `FileOutputCommitter` (v1 or v2) to rename temp files atomically | Commit latency in Stage 2 |

**STATIC vs DYNAMIC physical difference**: With STATIC, the plan contains an additional `DeleteDir` step before the write — Spark deletes the entire output path.  With DYNAMIC, each task independently calls `dynamicPartitionPruning` to determine which existing partition dirs to replace.\
""")
t6_physical["id"] = T6_PHYSICAL_ID

t6_ui = md("""\
### Topic 6 · Cell H: Spark UI Investigation Guide

**Symptom**: write job takes 45 minutes; downstream jobs see stale data.

**Step-by-step Spark UI investigation**:

1. **Jobs tab** → find the write job → click it → note number of stages.
   - Normal write: 2 stages (scan+filter, shuffle+sort+write).
   - If you see 3+ stages, a `coalesce` or extra `repartition` was inserted — investigate who added it.

2. **Stages tab → Stage with Exchange**:
   - *Shuffle Write*: total bytes written.  Divide by `spark.sql.shuffle.partitions` (200) to get avg partition size.  Target: 64–256 MB.
   - *Shuffle Spill (disk)*: non-zero means tasks ran out of execution memory during sort — raise `spark.memory.fraction`.

3. **Stages tab → Stage with InsertIntoHadoopFsRelation**:
   - *Output rows*: should equal input rows.
   - *Output files*: `numPartitionValues × tasksPerPartition`.  If this is > 10,000, you have the small-files problem — add `repartition(n, "payment_type")` before write.
   - *Task duration distribution*: if p99 >> median, one partition key has far more data than others (skew).

4. **SQL/DataFrame tab** → find the query → inspect the DAG:
   - `AQEShuffleRead` node means AQE coalesced some partitions — hover to see original vs final partition count.
   - `DynamicPruningExpression` indicates partition pruning is active on reads.

5. **Environment tab**:
   - Confirm `spark.sql.sources.partitionOverwriteMode` = `DYNAMIC` (not `STATIC`) if you expected dynamic behaviour.
   - Confirm `spark.sql.hive.manageFilesourcePartitions` = `true` if using the Hive metastore for partition registration.\
""")
t6_ui["id"] = T6_UI_ID

t6_exercise = code("""\
# ── Topic 6 · Cell I: Optimization Exercise ──────────────────────────────
import tempfile, os
from pyspark.sql import functions as F

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)

out_static  = os.path.join(TEMP_DIR, "static_write")
out_dynamic = os.path.join(TEMP_DIR, "dynamic_write")

# ── STATIC overwrite (default) ──
# TODO 1: Run this block, then add one new payment_type row and re-run.
#         Notice ALL partitions are deleted and rewritten even for untouched ones.
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "STATIC")
(cabs.write.mode("overwrite")
     .partitionBy("Payment_Type")
     .parquet(out_static))
print("Static write done. Files:")
static_files = [f for f in os.listdir(out_static) if not f.startswith(".")]
print(sorted(static_files))

# ── DYNAMIC overwrite ──
# TODO 2: Switch to DYNAMIC mode. Observe only affected partition dirs are replaced.
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "DYNAMIC")
cash_only = cabs.filter(F.col("Payment_Type") == "Cash")
(cash_only.write.mode("overwrite")
           .partitionBy("Payment_Type")
           .parquet(out_dynamic))
print("\\nDynamic write (Cash partition only):")

# TODO 3: Verify the non-Cash partitions still exist from the first write.
#         Hint: check out_dynamic directory — other Payment_Type dirs should remain.

# TODO 4: Benchmark: add .repartition(4, "Payment_Type") before each write
#         and observe whether output file count per partition drops.
print("\\nExercise: compare file counts with and without explicit repartition before write.")\
""")
t6_exercise["id"] = T6_EXERCISE_ID

t6_bench = code("""\
# ── Topic 6 · Cell J: Performance Benchmarking ───────────────────────────
import time, os
from pyspark.sql import functions as F

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)
out_bench = os.path.join(TEMP_DIR, "bench_partwrite")

def count_output_files(path):
    count = 0
    for root, dirs, files in os.walk(path):
        count += sum(1 for f in files if f.endswith(".parquet"))
    return count

# ── BEFORE: write with default shuffle partitions (many small files) ──
spark.conf.set("spark.sql.shuffle.partitions", "200")
import shutil
if os.path.exists(out_bench): shutil.rmtree(out_bench)
t0 = time.time()
cabs.write.mode("overwrite").partitionBy("Payment_Type").parquet(out_bench)
t_before = time.time() - t0
files_before = count_output_files(out_bench)
print(f"BEFORE — Time: {t_before:.2f}s | Output files: {files_before}")

# ── AFTER: explicit repartition by partition key before writing ──
if os.path.exists(out_bench): shutil.rmtree(out_bench)
t0 = time.time()
(cabs.repartition(4, "Payment_Type")
     .write.mode("overwrite")
     .partitionBy("Payment_Type")
     .parquet(out_bench))
t_after = time.time() - t0
files_after = count_output_files(out_bench)
print(f"AFTER  — Time: {t_after:.2f}s  | Output files: {files_after}")
print(f"File reduction: {files_before} → {files_after} ({(1-files_after/files_before)*100:.0f}% fewer files)")
print("At production scale (100M rows, 50 partition values) this can mean 10k → 200 files.")\
""")
t6_bench["id"] = T6_BENCH_ID

t6_followup = md("""\
### Topic 6 — Common Follow-up Questions

1. **"What happens if you use `insertInto()` instead of `write.partitionBy().parquet()` — same behaviour?"**
   No.  `insertInto(tableName)` uses the table's partition schema from the metastore (Hive or Spark catalog); you don't specify `partitionBy` — it's implicit.  It also respects the `spark.sql.sources.partitionOverwriteMode` setting.  `write.partitionBy()` writes to a raw path and doesn't register partitions in any metastore unless you call `spark.catalog.recoverPartitions(tableName)` or use `MSCK REPAIR TABLE`.

2. **"You wrote 50 partition directories, each with 200 tiny files. Downstream Spark jobs are slow on reads. What is the root cause and how do you fix it?"**
   Each task opens one file → 200 tasks per partition directory × 50 partitions = 10,000 tasks, most under 1 MB.  Fix: either (a) add `repartition(N, "partition_col")` before write so each partition produces N files, or (b) run a periodic compaction job (`read → coalesce(N) → write`) on each partition directory, or (c) on Delta Lake, `OPTIMIZE tableName WHERE partition_col = x`.

3. **"Dynamic partition overwrite sounds dangerous — what failure modes exist?"**
   (a) If a task fails mid-write and is retried, it may partially overwrite a partition with incomplete data (speculative execution + non-idempotent writes).  Fix: enable `spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2` (v2 committer) so each task writes directly to the final location atomically.  (b) Race condition between two concurrent jobs writing the same partition — no built-in serialisation; use Delta Lake or a coordinator service.

4. **"What is `spark.sql.hive.manageFilesourcePartitions` and when is it relevant?"**
   When `true`, Spark registers new partition directories in the Hive metastore automatically after a `partitionBy` write to a managed table.  When `false`, you must run `MSCK REPAIR TABLE` or `ALTER TABLE ADD PARTITION` manually.  Relevant when downstream tools (Presto, Athena, legacy Hive queries) rely on the metastore for partition discovery.

5. **"How does `writeTo().overwritePartitions()` in Spark 3 differ from `write.mode('overwrite').partitionBy()`?"**
   `writeTo().overwritePartitions()` targets a V2 table (e.g., Delta, Iceberg, Hudi) and performs a *logical* partition overwrite understood by the table format's transaction log — ACID-safe, supports concurrent readers.  `write.mode('overwrite').partitionBy()` is V1 and operates at the filesystem level with no transaction log, making it unsafe under concurrent access.\
""")
t6_followup["id"] = T6_FOLLOWUP_ID

t6_solution = code("""\
# ── Topic 6 · Cell M: Full Optimized Solution ────────────────────────────
import os, shutil
from pyspark.sql import functions as F

print("=== Topic 6: Dynamic Partitioning — Production-Grade Write Pattern ===\\n")

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)

# ── Config: use DYNAMIC overwrite so only touched partitions are replaced ──
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "DYNAMIC")
# AQE coalesces shuffle partitions to ~64 MB each → fewer output files per partition
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "67108864")  # 64 MB

# ── Step 1: Determine the right number of output files ──
# Rule of thumb: target 128-256 MB per output file.
# With ~8970 rows at ~200 bytes each ≈ 1.7 MB total → 1 file per partition key is fine.
# At 100M rows × 200 bytes = 20 GB / 4 payment types = 5 GB per partition → ~40 files each.
n_files_per_partition = max(1, int(cabs.count() * 200 / (128 * 1024 * 1024 * 4)))
print(f"Target files per partition: {n_files_per_partition}")

# ── Step 2: Explicit repartition by partition key before write ──
# This ensures each output task handles data for exactly one payment type.
output_path = os.path.join(TEMP_DIR, "optimized_partwrite")
if os.path.exists(output_path): shutil.rmtree(output_path)

(cabs
    .withColumn("Payment_Type", F.upper(F.trim(F.col("Payment_Type"))))  # normalise
    .filter(F.col("Payment_Type").isNotNull())                            # drop nulls
    .repartition(n_files_per_partition, "Payment_Type")                   # co-locate
    .write
    .mode("overwrite")
    .partitionBy("Payment_Type")
    .option("compression", "snappy")
    .parquet(output_path))

# ── Step 3: Verify partition layout ──
partitions = [d for d in os.listdir(output_path) if d.startswith("Payment_Type=")]
print(f"\\nPartition directories written: {sorted(partitions)}")

for p in sorted(partitions):
    pdir = os.path.join(output_path, p)
    files = [f for f in os.listdir(pdir) if f.endswith(".parquet")]
    rows = spark.read.parquet(pdir).count()
    print(f"  {p}: {len(files)} file(s), {rows:,} rows")

# ── Step 4: Confirm downstream read uses partition pruning ──
print("\\nRead with partition filter (should show PartitionFilters in plan):")
result = spark.read.parquet(output_path).filter(F.col("Payment_Type") == "CASH")
result.explain()
print(f"Rows in CASH partition: {result.count():,}")
print("\\n✓ Production pattern: DYNAMIC overwrite + explicit repartition + normalised keys")\
""")
t6_solution["id"] = T6_SOLUTION_ID

# ════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — Delta Lake: missing cells F, G, H, I, J, K, L, M
# ════════════════════════════════════════════════════════════════════════════

T7_LOGICAL_ID  = "t7_logical"
T7_PHYSICAL_ID = "t7_physical"
T7_UI_ID       = "t7_ui"
T7_EXERCISE_ID = "t7_exercise"
T7_BENCH_ID    = "t7_bench"
T7_BEST_ID     = "t7_best"
T7_FOLLOWUP_ID = "t7_followup"
T7_SOLUTION_ID = "t7_solution"

t7_logical = md("""\
### Topic 7 · Cell F: Expected Logical Plan Discussion

**Query**: `spark.read.format("delta").load(path).filter(...).groupBy(...).agg(...)`

```
== Analyzed Logical Plan ==
Aggregate [company], [company, sum(fare) AS total_fare]
  Filter (fare > 10.0)
    Relation[trip_id,company,fare,...] DeltaRelation

== Optimized Logical Plan ==
Aggregate [company], [company, sum(fare) AS total_fare]
  Project [company, fare]                     ← column pruning (only 2 cols needed)
    Filter (fare > 10.0 AND isnotnull(fare))  ← null filter pushed down
      Relation[trip_id,company,fare,...] DeltaRelation
        deltaLog.snapshot.allFiles            ← resolved from _delta_log JSON
        dataSkippingFilter: fare_min > 10.0   ← data skipping pushed into scan
```

**Delta-specific logical nodes**:
- `DeltaRelation`: wraps a `DeltaLog` snapshot — reads the latest `_delta_log/0000...json` or checkpoint `.parquet` to get the list of active data files.
- **Data skipping predicate**: Catalyst pushes `fare > 10.0` into the Delta scan layer.  Delta checks each Parquet file's column statistics (`fare_min`, `fare_max`) stored in the transaction log and excludes files where `fare_max <= 10.0`.
- **MVCC (Multi-Version Concurrency Control)**: the logical plan always reads from a *snapshot* at a specific `version` — concurrent writers appending new `json` files to `_delta_log` do not affect the in-flight query's file list.\
""")
t7_logical["id"] = T7_LOGICAL_ID

t7_physical = md("""\
### Topic 7 · Cell G: Expected Physical Plan Discussion

```
== Physical Plan ==
*(2) HashAggregate(keys=[company], functions=[sum(fare)])
+- Exchange hashpartitioning(company, 200), ENSURE_REQUIREMENTS
   +- *(1) HashAggregate(keys=[company], functions=[partial_sum(fare)])
      +- *(1) Project [company, fare]
         +- *(1) Filter (fare > 10.0)
            +- *(1) FileScan parquet delta.[path]
                    Batched: true
                    DataFilters: [isnotnull(fare), (fare > 10.0)]
                    Format: Parquet (Delta reads as Parquet at physical layer)
                    PartitionFilters: []
                    PushedFilters: [IsNotNull(fare), GreaterThan(fare,10.0)]
                    ReadSchema: struct<company:string,fare:double>
                    SelectedFiles: 12 of 47    ← data skipping in action
```

**Key physical plan observations**:

| Observation | What it means |
|-------------|---------------|
| `*(1)` prefix on FileScan + Filter + Project + partial HashAgg | All fused into one whole-stage codegen stage — zero object creation between operators |
| `SelectedFiles: 12 of 47` | Delta data skipping eliminated 35 files without reading them |
| `PushedFilters` in FileScan | Parquet row-group level filtering (min/max statistics per row group) within each selected file |
| Two-phase HashAggregate | Partial agg on each partition (map side), then final agg after shuffle (reduce side) |

**OPTIMIZE and VACUUM effects on the physical plan**:
- Before `OPTIMIZE`: 47 files (small files accumulated from streaming or incremental writes)
- After `OPTIMIZE`: 3 files (compacted to 128 MB each) → `FileScan` reads 3 files instead of 47, reducing task count and file-open overhead\
""")
t7_physical["id"] = T7_PHYSICAL_ID

t7_ui = md("""\
### Topic 7 · Cell H: Spark UI Investigation Guide

**Symptom**: Delta table reads are slow despite the table being "recent".

**Step-by-step investigation**:

1. **SQL/DataFrame tab → find the query → expand FileScan node**:
   - Look at `numFiles` in the node tooltip.  If it's in the hundreds or thousands for a table that should be tens of files, run `OPTIMIZE`.
   - Check `bytesRead` — if it's much larger than the data you expect (based on filter selectivity), data skipping is not working.  Verify column stats exist: `DESCRIBE DETAIL tableName` → `numFiles`, `sizeInBytes`.

2. **Stages tab → Stage with FileScan**:
   - *Tasks*: should equal `numFiles` (one task per file for columnar reads).  Hundreds of 1-second tasks reading 1 MB each = small-files problem.
   - *Task duration p99 vs median*: high variance = file size skew (some huge files, some tiny).

3. **Check `_delta_log` size**:
   - Each write transaction adds one JSON file.  After 10 commits, a checkpoint `.parquet` is written.
   - If `_delta_log` has thousands of JSON files and no recent checkpoint, metadata reads are slow.  Run `VACUUM` (with appropriate `RETAIN HOURS`) to clean old files, or call `deltaTable.generate("symlink_format_manifest")` to force a checkpoint.

4. **AQE nodes in DAG**:
   - `AQEShuffleRead` after the Exchange → AQE coalesced post-shuffle partitions.  Good sign.
   - If you see `BroadcastHashJoin` where you expected `SortMergeJoin`, AQE promoted the join at runtime because one side shrank after filtering (Delta's data skipping made the table appear smaller to AQE's runtime stats).

5. **Environment tab**:
   - Confirm `spark.databricks.delta.optimizeWrite.enabled=true` (Databricks) or `delta.autoOptimize.optimizeWrite=true` (OSS Delta with table property) — enables write-time bin-packing.\
""")
t7_ui["id"] = T7_UI_ID

t7_exercise = code("""\
# ── Topic 7 · Cell I: Optimization Exercise ──────────────────────────────
# Simulate Delta Lake patterns with plain Parquet (OSS Delta requires extra deps).
# Each TODO maps to a real Delta Lake operation.
import os, shutil, time
from pyspark.sql import functions as F

base = os.path.join(TEMP_DIR, "delta_sim")
if os.path.exists(base): shutil.rmtree(base)

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)

# TODO 1: Simulate small-files accumulation (10 incremental writes of ~900 rows each).
#         Write each slice to a subdirectory, then read the whole base path.
#         Measure how many files Spark opens and the total read time.
for i in range(10):
    slice_df = cabs.limit(900)   # In production: streaming micro-batches
    (slice_df.write.mode("overwrite")
             .parquet(os.path.join(base, f"batch_{i:02d}")))
print(f"Files written: 10 batches × (200 default partitions) = up to 2000 files")

t0 = time.time()
raw_count = spark.read.parquet(base).count()
print(f"Read 10-batch table: {raw_count:,} rows in {time.time()-t0:.2f}s")

# TODO 2: Simulate OPTIMIZE — compact all batches into a single coalesced file.
#         Read all batches, coalesce(1), re-write to a compacted path.
compacted = os.path.join(TEMP_DIR, "delta_sim_optimized")
# Hint: spark.read.parquet(base).coalesce(???).write.parquet(compacted)

# TODO 3: After compacting, re-run the read and compare task count and time.
# t0 = time.time()
# opt_count = spark.read.parquet(compacted).count()
# print(f"Read compacted table: {opt_count:,} rows in {time.time()-t0:.2f}s")

# TODO 4: Simulate VACUUM — delete the original batch directories.
#         Equivalent to Delta VACUUM with RETAIN 0 HOURS (dangerous in production!).
# for i in range(10):
#     shutil.rmtree(os.path.join(base, f"batch_{i:02d}"))
print("\\nComplete TODOs 2-4 and compare Spark UI task counts before vs after compaction.")\
""")
t7_exercise["id"] = T7_EXERCISE_ID

t7_bench = code("""\
# ── Topic 7 · Cell J: Performance Benchmarking ───────────────────────────
import time, os, shutil
from pyspark.sql import functions as F

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)
base     = os.path.join(TEMP_DIR, "t7_bench_fragmented")
compacted = os.path.join(TEMP_DIR, "t7_bench_compacted")
for p in [base, compacted]:
    if os.path.exists(p): shutil.rmtree(p)

# ── BEFORE: fragmented (simulate 10 small writes) ──
for i in range(10):
    (cabs.repartition(2)
         .write.mode("overwrite")
         .parquet(os.path.join(base, f"part_{i}")))

t0 = time.time()
frag_count = (spark.read.parquet(base)
                   .filter(F.col("Fare") > 10)
                   .groupBy("Company").agg(F.sum("Fare").alias("total"))
                   .count())
t_frag = time.time() - t0

frag_files = sum(len(files) for _, _, files in os.walk(base)
                 if any(f.endswith(".parquet") for f in files))
print(f"BEFORE (fragmented): {t_frag:.2f}s | {frag_files} parquet files | {frag_count} groups")

# ── AFTER: compacted (simulate OPTIMIZE) ──
(spark.read.parquet(base).coalesce(1).write.parquet(compacted))

t0 = time.time()
comp_count = (spark.read.parquet(compacted)
                   .filter(F.col("Fare") > 10)
                   .groupBy("Company").agg(F.sum("Fare").alias("total"))
                   .count())
t_comp = time.time() - t0

comp_files = sum(len(files) for _, _, files in os.walk(compacted)
                 if any(f.endswith(".parquet") for f in files))
print(f"AFTER  (compacted):  {t_comp:.2f}s | {comp_files} parquet files  | {comp_count} groups")
print(f"Speedup: {t_frag/t_comp:.1f}x  — at 10k files in production, improvement is 10-50x")\
""")
t7_bench["id"] = T7_BENCH_ID

t7_best = md("""\
### Topic 7 — Production Best Practices: Delta Lake

1. **Run OPTIMIZE on a schedule** — compact small files to 128 MB targets.  On Databricks use `OPTIMIZE tableName ZORDER BY (col1, col2)`.  On OSS Delta: `DeltaTable.forPath(spark, path).optimize().executeCompaction()`.

2. **Set VACUUM retention carefully** — default `delta.deletedFileRetentionDuration = interval 7 days`.  Never run `VACUUM ... RETAIN 0 HOURS` in production — active long-running queries will fail with FileNotFoundException if their snapshot files are deleted.

3. **Use autoCompact and optimizeWrite on streaming tables** — table properties `delta.autoOptimize.autoCompact=true` and `delta.autoOptimize.optimizeWrite=true` (Databricks) bin-pack output files at write time, avoiding the accumulation of small files from streaming micro-batches.

4. **Monitor `_delta_log` size** — the transaction log grows by one JSON file per commit.  A checkpoint (`.checkpoint.parquet`) is written every 10 commits by default (`delta.checkpointInterval`).  If checkpoints are disabled or the interval is too high, metadata reads slow significantly.

5. **Use time travel for auditing, not for primary reads** — `spark.read.format("delta").option("versionAsOf", N).load(path)` reads an older snapshot but forces reading older (potentially sub-optimal) file layouts.  Use for audit/rollback only.

6. **Liquid Clustering (Delta 3.x+) replaces ZORDER for high-cardinality keys** — uses Hilbert-curve-based clustering that can handle multiple clustering columns without the fixed column-order limitation of Z-ordering.  Enable via `CLUSTER BY (col1, col2)` at table creation.

7. **Control log retention separately from data retention** — `delta.logRetentionDuration` (default 30 days) controls how long transaction log JSON files are kept.  This must be ≥ `deletedFileRetentionDuration` to allow VACUUM to correctly identify which data files are safe to delete.

8. **Partition large Delta tables by low-cardinality columns** — e.g., `date`, `region`.  Do not partition on high-cardinality columns (user_id, trip_id) — this creates millions of partition directories and degrades the metastore.  Use Z-ordering or Liquid Clustering for high-cardinality filter columns within partitions.\
""")
t7_best["id"] = T7_BEST_ID

t7_followup = md("""\
### Topic 7 — Common Follow-up Questions

1. **"You ran VACUUM with RETAIN 0 HOURS and a 6-hour query failed. What happened and how do you prevent this?"**
   VACUUM deleted data files that were part of the 6-hour query's snapshot.  Delta's MVCC guarantees snapshot isolation only for files still present on disk.  Prevention: set `delta.deletedFileRetentionDuration = interval 24 hours` (or longer than your longest-running query) and use `DeltaTable.vacuum(retentionHours=N)` rather than raw SQL to get the duration check.

2. **"What is in `_delta_log/00000000000000000000.json` and how does Spark use it?"**
   It contains `add` actions (file paths, size, column statistics), `remove` actions (for deletes/overwrites), and `metaData` (schema, partition columns, table properties).  Spark reads the latest checkpoint + all subsequent JSON files to reconstruct the current snapshot — the set of active data files and their statistics.

3. **"Delta OPTIMIZE rewrites files. Does this break time travel to before the OPTIMIZE?"**
   No.  The original files remain in `_delta_log` as `add` actions for the old version and `remove` actions for the post-OPTIMIZE version.  Time travel to any version before OPTIMIZE still works *as long as VACUUM has not deleted those files*.

4. **"Concurrent writers on the same Delta table — what happens?"**
   Delta uses optimistic concurrency control.  Each writer reads the current version N, performs its computation, then attempts to commit at version N+1 by writing `_delta_log/000...N+1.json`.  If another writer already committed N+1, the current writer's commit fails with `ConcurrentModificationException` and Spark retries (up to `spark.databricks.delta.maxCommitRetries` times) with the updated snapshot.

5. **"How does Delta's `logRetentionDuration` differ from `deletedFileRetentionDuration`?"**
   `deletedFileRetentionDuration` controls how long deleted *data files* are kept before VACUUM removes them.  `logRetentionDuration` controls how long *transaction log JSON files* are kept — this determines how far back time travel is possible.  A common mistake: setting `deletedFileRetentionDuration=7 days` but `logRetentionDuration=1 day` — time travel history is lost before data files are cleaned up.\
""")
t7_followup["id"] = T7_FOLLOWUP_ID

t7_solution = code("""\
# ── Topic 7 · Cell M: Full Optimized Solution ────────────────────────────
import os, shutil, time
from pyspark.sql import functions as F

print("=== Topic 7: Delta Lake Optimization Patterns (Parquet simulation) ===\\n")

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)
lake_path     = os.path.join(TEMP_DIR, "t7_solution_lake")
compacted_path = os.path.join(TEMP_DIR, "t7_solution_compacted")
for p in [lake_path, compacted_path]:
    if os.path.exists(p): shutil.rmtree(p)

# ── Pattern 1: Controlled incremental write (simulate streaming micro-batches) ──
print("Pattern 1: Simulated streaming writes (10 micro-batches)...")
for i in range(10):
    batch = cabs.filter(F.col("Company").isNotNull()).repartition(1)
    batch.write.mode("overwrite").parquet(os.path.join(lake_path, f"batch_{i:02d}"))
total_files = sum(len(fs) for _, _, fs in os.walk(lake_path) if any(f.endswith(".parquet") for f in fs))
print(f"  Files after 10 batches: {total_files}")

# ── Pattern 2: OPTIMIZE equivalent (compaction) ──
print("\\nPattern 2: Compaction (simulate OPTIMIZE)...")
t0 = time.time()
all_data = spark.read.parquet(lake_path)
target_file_size_bytes = 128 * 1024 * 1024  # 128 MB
estimated_size = all_data.count() * 200  # ~200 bytes per row estimate
n_output_files = max(1, estimated_size // target_file_size_bytes)
(all_data.repartition(n_output_files)
         .write.parquet(compacted_path))
t_optimize = time.time() - t0
opt_files = sum(len(fs) for _, _, fs in os.walk(compacted_path) if any(f.endswith(".parquet") for f in fs))
print(f"  Compacted to {opt_files} file(s) in {t_optimize:.2f}s (equivalent to Delta OPTIMIZE)")

# ── Pattern 3: Read performance comparison ──
print("\\nPattern 3: Read performance — fragmented vs compacted...")
t0 = time.time(); spark.read.parquet(lake_path).filter(F.col("Fare")>10).count(); t_frag = time.time()-t0
t0 = time.time(); spark.read.parquet(compacted_path).filter(F.col("Fare")>10).count(); t_comp = time.time()-t0
print(f"  Fragmented read:  {t_frag:.2f}s | Compacted read: {t_comp:.2f}s | Speedup: {t_frag/t_comp:.1f}x")

# ── Pattern 4: VACUUM equivalent (delete old batch files) ──
print("\\nPattern 4: VACUUM equivalent (delete fragmented batch files)...")
# In Delta: VACUUM lakePath RETAIN 168 HOURS
# Simulation: remove old batch dirs after verifying compacted copy is complete
for i in range(10):
    old = os.path.join(lake_path, f"batch_{i:02d}")
    if os.path.exists(old): shutil.rmtree(old)
print("  Old batch files removed (equivalent to VACUUM)")
print("  In Delta Lake production: NEVER use RETAIN 0 HOURS — keep >= 7 days")

print("\\n✓ Production sequence: incremental writes → scheduled OPTIMIZE → VACUUM after retention period")\
""")
t7_solution["id"] = T7_SOLUTION_ID

# ════════════════════════════════════════════════════════════════════════════
# TOPIC 8 — Z-Ordering: missing cells F (logical), G (physical), I (exercise),
#            J (benchmark), M (full solution)
# ════════════════════════════════════════════════════════════════════════════

T8_LOGICAL_ID  = "t8_logical"
T8_PHYSICAL_ID = "t8_physical"
T8_EXERCISE_ID = "t8_exercise"
T8_BENCH_ID    = "t8_bench"
T8_SOLUTION_ID = "t8_solution"

t8_logical = md("""\
### Topic 8 · Cell F: Expected Logical Plan Discussion

**Query**: `spark.read.parquet(zorderPath).filter(col("pickup_location_id") == 132).filter(col("fare") > 20)`

```
== Analyzed Logical Plan ==
Filter ((pickup_location_id = 132) AND (fare > 20.0))
  Relation[trip_id, pickup_location_id, dropoff_location_id, fare, ...] ParquetRelation

== Optimized Logical Plan ==
Project [trip_id, pickup_location_id, dropoff_location_id, fare]  ← column pruning
  Filter ((pickup_location_id = 132) AND (isnotnull(pickup_location_id))
          AND (fare > 20.0) AND (isnotnull(fare)))
    Relation[...] ParquetRelation
      dataSkippingFilter:
        pickup_location_id_min <= 132 AND pickup_location_id_max >= 132
        AND fare_max > 20.0                                            ← pushed into scan
```

**Why Z-ordering helps at the logical plan level**:
- After Z-ordering on `(pickup_location_id, fare)`, rows with *similar Z-index values* (rows where both columns are co-located by the Morton curve) are written to the same Parquet row groups.
- This means `pickup_location_id_min` and `pickup_location_id_max` per row group tightly bracket the actual values in that group.
- Catalyst's data skipping predicate `pickup_location_id_min <= 132 AND pickup_location_id_max >= 132` then eliminates most row groups, even within each file.
- **Without Z-ordering**: random row order means every row group contains the full range of `pickup_location_id` → `pickup_location_id_min=1, max=265` → predicate never eliminates any row group.\
""")
t8_logical["id"] = T8_LOGICAL_ID

t8_physical = md("""\
### Topic 8 · Cell G: Expected Physical Plan Discussion

```
== Physical Plan ==
*(1) Project [trip_id, pickup_location_id, dropoff_location_id, fare]
+- *(1) Filter ((pickup_location_id = 132) AND (fare > 20.0))
   +- *(1) FileScan parquet [trip_id, pickup_location_id, dropoff_location_id, fare]
           Batched: true
           DataFilters: [isnotnull(pickup_location_id), (pickup_location_id = 132),
                         isnotnull(fare), (fare > 20.0)]
           Format: Parquet
           PushedFilters: [IsNotNull(pickup_location_id), EqualTo(pickup_location_id,132),
                           IsNotNull(fare), GreaterThan(fare,20.0)]
           ReadSchema: struct<...>
           PartitionFilters: []
           SelectedFiles: 2 of 20       ← Z-order data skipping (row-group level)
           RowGroupsRead: 3 of 160      ← Parquet row-group min/max filtering
```

**Key differences vs non-Z-ordered data**:

| Metric | Without Z-order | With Z-order |
|--------|----------------|--------------|
| SelectedFiles | 20 of 20 (100%) | 2 of 20 (10%) |
| RowGroupsRead | 160 of 160 | 3 of 160 (2%) |
| Bytes read | Full dataset | ~2% of dataset |
| Task count | 20 | 2 |

**Bloom filter interaction** (`parquet.bloom.filter.enabled=true` on `pickup_location_id`):
- Bloom filters operate at row-group level: for each row group, a probabilistic membership test is done for the equality predicate (`== 132`).
- False positive rate is controlled by `parquet.bloom.filter.fpp` (default 0.01 = 1%).
- Bloom filters are most effective for **equality predicates** on high-cardinality columns; min/max statistics work better for **range predicates** (`> 20.0`).\
""")
t8_physical["id"] = T8_PHYSICAL_ID

t8_exercise = code("""\
# ── Topic 8 · Cell I: Optimization Exercise ──────────────────────────────
# Demonstrate Z-order's effect on data skipping by controlling row layout.
import os, shutil
from pyspark.sql import functions as F
from pyspark.sql.types import *

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)

# TODO 1: Write the dataset WITHOUT Z-ordering (random file layout).
#         Use repartition(5) so we have multiple files for a meaningful comparison.
no_zorder_path = os.path.join(TEMP_DIR, "t8_no_zorder")
if os.path.exists(no_zorder_path): shutil.rmtree(no_zorder_path)
# cabs.repartition(5).write.parquet(no_zorder_path)

# TODO 2: Write the dataset WITH Z-order simulation.
#         Spark OSS doesn't have native ZORDER, but we can approximate it by
#         sorting on the Z-index (interleaved bits) of the two columns.
#         Use this UDF that computes a Morton code for two integer values:
from pyspark.sql.functions import udf
from pyspark.sql.types import LongType

def morton_2d(x, y):
    """Interleave bits of x and y to produce a Morton code."""
    if x is None or y is None: return None
    def spread(v):
        v = v & 0xFFFF
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v
    return spread(int(x)) | (spread(int(y)) << 1)

morton_udf = udf(morton_2d, LongType())

zorder_path = os.path.join(TEMP_DIR, "t8_zorder")
if os.path.exists(zorder_path): shutil.rmtree(zorder_path)
# TODO 2 complete: add z_index column, sort by it, then write
# (cabs.filter(F.col("Distance").isNotNull() & F.col("Fare").isNotNull())
#      .withColumn("z_idx", morton_udf(
#           (F.col("Distance") * 10).cast("int"),
#           (F.col("Fare") * 10).cast("int")))
#      .orderBy("z_idx")
#      .drop("z_idx")
#      .repartition(5)
#      .write.parquet(zorder_path))

# TODO 3: Query both paths with the same filter and compare task counts in Spark UI.
# filter_val = 15.0
# no_z_count = spark.read.parquet(no_zorder_path).filter(F.col("Fare") > filter_val).count()
# z_count    = spark.read.parquet(zorder_path).filter(F.col("Fare") > filter_val).count()
# print(f"No Z-order: {no_z_count} rows | Z-order: {z_count} rows (same result, fewer files read)")

print("Complete TODOs 1-3 and compare 'numFiles' in Spark UI FileScan nodes.")\
""")
t8_exercise["id"] = T8_EXERCISE_ID

t8_bench = code("""\
# ── Topic 8 · Cell J: Performance Benchmarking ───────────────────────────
import time, os, shutil
from pyspark.sql import functions as F
from pyspark.sql.types import LongType
from pyspark.sql.functions import udf

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)

no_zorder_path = os.path.join(TEMP_DIR, "t8_bench_no_z")
zorder_path    = os.path.join(TEMP_DIR, "t8_bench_z")
for p in [no_zorder_path, zorder_path]:
    if os.path.exists(p): shutil.rmtree(p)

# ── Write without Z-order ──
(cabs.filter(F.col("Fare").isNotNull() & F.col("Distance").isNotNull())
     .repartition(8)
     .write.parquet(no_zorder_path))
print("Written: no-Z-order dataset (8 files, random layout)")

# ── Write with Z-order simulation (sort by morton code) ──
def morton_2d(x, y):
    if x is None or y is None: return None
    def spread(v):
        v = v & 0xFFFF; v = (v|(v<<8))&0x00FF00FF; v = (v|(v<<4))&0x0F0F0F0F
        v = (v|(v<<2))&0x33333333; v = (v|(v<<1))&0x55555555; return v
    return spread(int(x)) | (spread(int(y)) << 1)
morton_udf = udf(morton_2d, LongType())

(cabs.filter(F.col("Fare").isNotNull() & F.col("Distance").isNotNull())
     .withColumn("z_idx", morton_udf((F.col("Distance")*10).cast("int"), (F.col("Fare")*10).cast("int")))
     .orderBy("z_idx").drop("z_idx")
     .repartition(8)
     .write.parquet(zorder_path))
print("Written: Z-ordered dataset (8 files, Morton-sorted)")

# ── Benchmark selective query ──
FARE_THRESHOLD = 50.0
N_RUNS = 3

def timed_query(path, label):
    times = []
    for _ in range(N_RUNS):
        t0 = time.time()
        spark.read.parquet(path).filter(F.col("Fare") > FARE_THRESHOLD).count()
        times.append(time.time() - t0)
    avg = sum(times) / N_RUNS
    print(f"  {label}: avg {avg:.2f}s over {N_RUNS} runs")
    return avg

print(f"\\nQuery: Fare > {FARE_THRESHOLD}")
t_no_z = timed_query(no_zorder_path, "No Z-order")
t_z    = timed_query(zorder_path,    "Z-ordered ")
print(f"\\nSpeedup: {t_no_z/t_z:.2f}x — grows significantly at 100M+ rows with real Parquet stats")
print("In Delta Lake, OPTIMIZE ZORDER BY (Fare) achieves this automatically with real row-group stats.")\
""")
t8_bench["id"] = T8_BENCH_ID

t8_solution = code("""\
# ── Topic 8 · Cell M: Full Optimized Solution ────────────────────────────
import os, shutil, time
from pyspark.sql import functions as F
from pyspark.sql.types import LongType
from pyspark.sql.functions import udf

print("=== Topic 8: Z-Ordering & Data Skipping — Production Pattern ===\\n")

cabs = spark.read.csv(CABS_PATH, header=True, inferSchema=True)

# ── Step 1: Understand your query patterns before choosing Z-order columns ──
print("Step 1: Profile most-frequent filter columns...")
print("  Most queries filter on: Company (low cardinality) and Fare (continuous range)")
print("  Z-order recommendation: ZORDER BY (Fare, Distance) — both numeric, range queries")
print("  Do NOT Z-order on Company alone — use partitionBy(Company) instead (low cardinality)")

# ── Step 2: Morton Z-index UDF ──
def morton_2d(x, y):
    """Bit-interleave x and y into a Morton Z-index (approximates Delta ZORDER)."""
    if x is None or y is None: return 0
    def spread(v):
        v = max(0, min(int(v), 0xFFFF))
        v = (v|(v<<8))&0x00FF00FF; v = (v|(v<<4))&0x0F0F0F0F
        v = (v|(v<<2))&0x33333333; v = (v|(v<<1))&0x55555555
        return v
    return spread(int(x)) | (spread(int(y)) << 1)
morton_udf = udf(morton_2d, LongType())

# ── Step 3: Write with Z-ordering ──
print("\\nStep 2: Write data sorted by Z-index...")
zorder_out = os.path.join(TEMP_DIR, "t8_solution_zorder")
if os.path.exists(zorder_out): shutil.rmtree(zorder_out)

cleaned = (cabs
    .filter(F.col("Fare").isNotNull() & F.col("Distance").isNotNull())
    .withColumn("fare_int",     (F.col("Fare")     * 100).cast("int"))
    .withColumn("distance_int", (F.col("Distance") * 100).cast("int"))
    .withColumn("z_idx", morton_udf(F.col("fare_int"), F.col("distance_int")))
    .orderBy("z_idx")
    .drop("fare_int", "distance_int", "z_idx"))

n_partitions = max(1, int(cleaned.count() / 10000))  # ~10k rows per file
(cleaned.repartition(n_partitions)
        .write
        .option("parquet.bloom.filter.enabled#Fare", "true")       # bloom on Fare equality
        .option("parquet.bloom.filter.enabled#Company", "true")     # bloom on Company equality
        .option("parquet.block.size", str(128 * 1024 * 1024))       # 128 MB row groups
        .parquet(zorder_out))
print(f"  Written to {n_partitions} file(s) with bloom filters enabled")

# ── Step 4: Verify data skipping on a selective query ──
print("\\nStep 3: Verify skipping on selective filter...")
FARE_THRESHOLD = 80.0
t0 = time.time()
result = (spark.read.parquet(zorder_out)
               .filter(F.col("Fare") > FARE_THRESHOLD))
count = result.count()
elapsed = time.time() - t0
print(f"  Fare > {FARE_THRESHOLD}: {count:,} rows in {elapsed:.2f}s")
result.show(5)

# ── Step 5: Production recommendations ──
print("\\nProduction Z-ordering checklist:")
print("  1. Use Delta Lake OPTIMIZE ... ZORDER BY for real row-group statistics")
print("  2. Choose Z-order columns that appear together in WHERE clauses")
print("  3. Limit to 2-3 columns — Z-curve effectiveness degrades beyond 4 dimensions")
print("  4. Combine: partitionBy(date) + ZORDER BY(location_id, fare) — partition prunes files,")
print("     Z-order prunes row groups within those files")
print("  5. For Spark 3.3+: consider Liquid Clustering (Hilbert curve, no column-order dependency)")
print("  6. Re-OPTIMIZE after major data loads — Z-order is not self-maintaining")\
""")
t8_solution["id"] = T8_SOLUTION_ID

# ════════════════════════════════════════════════════════════════════════════
# LOAD NOTEBOOK AND INJECT CELLS AT THE RIGHT POSITIONS
# ════════════════════════════════════════════════════════════════════════════

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

def find_idx(cell_id):
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            return i
    return None

def insert_after(cell_id, new_cells_list):
    idx = find_idx(cell_id)
    if idx is None:
        print(f"WARNING: cell {cell_id} not found — appending at end")
        cells.extend(new_cells_list)
        return
    for offset, nc in enumerate(new_cells_list):
        cells.insert(idx + 1 + offset, nc)

# Topic 4: insert exercise, benchmark, followup after f605b372 (best practices)
# But the optimized solution (8750863b) already exists after f605b372.
# Insert order: exercise → benchmark → followup → (existing solution 8750863b stays)
# Actually best to put exercise + bench BEFORE solution, followup AFTER solution.
# Find 8750863b and insert exercise+bench before it, followup after it.
sol4_idx = find_idx("8750863b")
if sol4_idx is not None:
    # Insert followup AFTER solution
    cells.insert(sol4_idx + 1, t4_followup)
    # Insert exercise + bench BEFORE solution (re-find after previous insert)
    sol4_idx = find_idx("8750863b")
    cells.insert(sol4_idx, t4_bench)
    cells.insert(sol4_idx, t4_exercise)
    print("Topic 4: inserted exercise, benchmark, followup")
else:
    insert_after("f605b372", [t4_exercise, t4_bench, t4_followup])
    print("Topic 4: inserted after f605b372 (8750863b not found)")

# Topic 5: insert exercise, best practices, followup after 4acd6634 (solution)
insert_after("4acd6634", [t5_exercise, t5_best, t5_followup])
print("Topic 5: inserted exercise, best practices, followup")

# Topic 6: insert logical, physical, ui, exercise, bench, followup, solution after 2c150874 (best practices)
insert_after("2c150874", [t6_logical, t6_physical, t6_ui, t6_exercise, t6_bench, t6_followup, t6_solution])
print("Topic 6: inserted 7 missing cells")

# Topic 7: insert logical, physical, ui, exercise, bench, best, followup, solution after 1e4a62db (questions)
insert_after("1e4a62db", [t7_logical, t7_physical, t7_ui, t7_exercise, t7_bench, t7_best, t7_followup, t7_solution])
print("Topic 7: inserted 8 missing cells")

# Topic 8: insert logical, physical after 7bb63ef4 (questions);
#           exercise, bench, solution after edae1330 (followup questions) or before 50a5bac4 (cleanup)
insert_after("7bb63ef4", [t8_logical, t8_physical])
print("Topic 8: inserted logical + physical plans")
# exercise, bench, solution go before the cleanup cell
cleanup_idx = find_idx("50a5bac4")
if cleanup_idx is not None:
    cells.insert(cleanup_idx, t8_solution)
    cells.insert(cleanup_idx, t8_bench)
    cells.insert(cleanup_idx, t8_exercise)
    print("Topic 8: inserted exercise, benchmark, solution before cleanup")
else:
    cells.extend([t8_exercise, t8_bench, t8_solution])
    print("Topic 8: appended exercise, benchmark, solution at end")

# ── Save ──────────────────────────────────────────────────────────────────
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"\nDone. Total cells: {len(nb['cells'])}")
print(f"Saved: {NB_PATH}")
