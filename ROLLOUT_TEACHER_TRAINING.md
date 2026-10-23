# Optional rollout-teacher training

This mode trains against the stored SWE-smith/R2E Teacher distributions. The existing
`next_token` objective remains the default. Choose
`--training_objective=teacher_top512_special_tail_kl` explicitly to use distillation.
The implementation does not select a production model, training budget or State horizon.

## Data and selection

The canonical dataset is listed in `Wiki/Projects/datasets.md` in the user's Codex Wiki;
read its `README.md` and `DATASET_CARD.md` before configuring a run. Inputs are its stored
token IDs. Do not render the raw conversations with a Student chat template or tokenize
them again. The Student must use the same ID mapping and the full 248,320-row vocabulary.

`scripts/build_rollout_teacher_index.py` creates a small index of pointers into those
existing ArrayRecords. It does not copy or recompute Teacher targets. Supply
`--dataset_root`, a new `--out_dir`, `--selection`, `--segment_length` and
`--segment_buckets` explicitly. The three selections are `correct`, `incorrect` and
`graded` (both). They require a completed attempt with a boolean correctness label;
truncated, timed-out and errored attempts are excluded. The original task-group train/val
split is preserved.

Each index binds one selection and one segmentation configuration. Changing either
creates a new pointer index, reusing the same stored tokens and logits. An existing
index is validated and reused; incompatible arguments are rejected. Startup checks
metadata and index hashes, not the raw-rollout tree or the entire Teacher-logit corpus.

## Whole chains and batch shapes

The segment length T is fixed within a run. Supply ascending segment-count buckets C
that cover every selected chain: a chain goes into the smallest fitting bucket.
Insufficient maximum capacity raises an error; chains are not silently cropped or
dropped. Segments retain chronological order, State carries within the chain, and a new
chain resets State. BPTT controls gradient detachment separately. It does not remove
Teacher targets at segment boundaries.

The existing `--batch_size` convention is a global number of segments. A bucket with C
segments therefore has `batch_size / C` global sample slots, also subject to process and
device divisibility. Before the model is built or a batch is read, training checks every
configured bucket against the actual device mesh with the same batch-sharding rule that
each batch must satisfy, so a later bucket cannot fail after earlier updates. Each JAX
process loads one iterator data shard of `batch_size / process_count` segments; the CLI
rejects `--iterator_dp_size` x `--iterator_fsdp_size` different from the process count,
because a shard without a loading process would silently miss its chains. All data shards
use the same bucket shape at a given step. Residual sample slots are padded and masked
rather than dropped or duplicated. Teacher rows have a fixed padded capacity per bucket;
context and padded positions get no KL.

Teacher runs require explicit `--grain_workers` and `--grain_read_prefetch_buffer_size`;
the pretraining defaults were sized for much smaller CE batches. Startup logs the largest
per-process host batch in bytes, computed from the bucket's sample slots, Teacher-row
capacity and the iterator's array dtypes (about 4.24 kB per Teacher row) without building
a batch. This size excludes transient copies and the other microbatches of an update,
which are read together before the first one trains. Multiply it by the batches that are
in flight, which depend on the Grain worker, worker-buffer and prefetch settings. A CPU
measurement of this iterator's Grain structure behind a stalled consumer counted 65 built
elements after one was consumed with 8 workers, 2 read threads, prefetch 4 and worker
buffer 1; 17 with 2 workers and 5 with 0 workers. These counts are operational evidence
for choosing the flags, not a measured GPU-run memory failure or a universal bound.

## Training, loss and validation

Use `scripts/train_text_pretrain.py` with the explicit objective above, a Statepassing
`--pretrain_mode`, `--train_index_path`, `--teacher_selection`,
`--teacher_segment_buckets`, matching `--seq_len=T`, and an explicit `--num_steps`.
The index contains both splits; do not supply a separate `--val_index_path` in this mode.
Model initialization, learning rate, batch size, accumulation, State/conv/RoPE/BPTT
settings, checkpointing and validation budget remain run choices through the existing
flags. Without `--init_hf_model_path` or `--model_id` the script trains its existing
randomly initialized default Student (12 layers, width 768, full vocabulary). Learning
rate 3e-4, `--warmup_tokens=150000000`, weight decay 0.1 and the cosine schedule are
inherited pretraining defaults unless specified. A production run must choose
initialization and these optimizer settings explicitly in its Objective-Contract.
Execute the builder and training under Slurm on compute nodes. Both entrypoints refuse
direct execution before importing NumPy, JAX or OmegaLAX unless `SLURM_JOB_ID` and
`SLURMD_NODENAME` are set and the node name matches the actual hostname.

The loss is grouped forward KL at temperature 1. It compares each unique Top512 ID,
each of the 33 special IDs not already present, and one aggregate class for the remaining
vocabulary. This is not exact full-vocabulary KL. There is no extra CE mixture or online
Teacher. A target at token position j uses Hidden position j-1, including boundary and
final targets. User/system/tool text provides context only.

Student vocabulary projection runs in bounded, rematerialized blocks of Teacher rows.
The normalizer covers the full Student vocabulary; the tail is calculated from its
complement rather than subtracting nearly equal probabilities. Gradients are weighted
by the total valid Teacher-target count across data shards and all microbatches of an
optimizer update.

Metrics include `teacher_kl`, its sum and target count, actual context-token count,
Teacher targets/s and context tokens/s. KL is not reported as CE perplexity. KL by family
or segment position and peak-GPU-memory telemetry are not implemented.

Validation restarts the same deterministic validation order at each evaluation;
`--val_steps` controls how much of that order is evaluated. The order is bucket-major from
the smallest segment bucket. Startup logs the batches, samples and targets of the whole
val split. If `--val_steps` is smaller than its batch count, startup warns that the
evaluated prefix favors shorter chains and is not whole-split validation, and gives the
full batch count. The chosen budget and order are not changed automatically; to measure
the whole split, set `--val_steps` to at least the logged batch count.

## Resume and switching modes

Checkpoints preserve the optimizer, RNG and Grain iterator position. The training
contract additionally binds objective, source/index identity, selection, segment
configuration and relevant run settings, including the global segment batch, gradient
accumulation steps and the number of iterator data shards (loading processes). Resume
with the same contract to continue the same data stream. A different
selection/objective/segmentation is a new experiment, rather than an unchanged resume.

The existing CE paths and dataset formats remain available through `next_token`.
Historical Teacher building/scoring code remains relevant for dataset creation; it is
not part of the Student training startup.

## Verification limits

CPU integration tests exercise the real serializer, iterator, model/loss, optimizer and
checkpoint path. A multi-GPU function/performance check is a separate run decision; CPU
tests do not establish GPU memory fit or throughput. The dataset's documented canonical
reconstruction and Teacher-engine numerical limitations still apply.
