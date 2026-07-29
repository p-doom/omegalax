"""Build the historical C4 tail bundle used by the 40k-to-44k experiments."""

from absl import app, flags

from omegalax.data.curriculum_tail import build_curriculum_c4_tail_bundle

FLAGS = flags.FLAGS

flags.DEFINE_string("curriculum_index_root", None, "Source curriculum bundle.", required=True)
flags.DEFINE_string("out_dir", None, "Output fixed-C tail bundle.", required=True)
flags.DEFINE_string("c4_iterator_state", None, "Checkpointed C4 Grain state JSON.", required=True)
flags.DEFINE_string("c8_iterator_state", None, "Checkpointed C8 Grain state JSON.", required=True)
flags.DEFINE_integer("c4_steps", 2316, "Historical C4 updates selected after checkpoint 40k.")
flags.DEFINE_integer("c8_steps", 1684, "Historical C8-equivalent updates selected next.")
flags.DEFINE_integer("trim_batch_size", 192, "Training segment batch size.")
flags.DEFINE_integer("trim_grad_accum_steps", 5, "Training gradient accumulation steps.")
flags.DEFINE_integer("seed", 0, "Historical Grain shuffle seed.")
flags.DEFINE_integer("records_per_shard", 100000, "ArrayRecord records per shard.")
flags.DEFINE_bool("overwrite", False, "Replace an existing output bundle.")


def main(_) -> None:
    out = build_curriculum_c4_tail_bundle(
        FLAGS.curriculum_index_root,
        FLAGS.out_dir,
        c4_iterator_state=FLAGS.c4_iterator_state,
        c8_iterator_state=FLAGS.c8_iterator_state,
        c4_steps=FLAGS.c4_steps,
        c8_steps=FLAGS.c8_steps,
        trim_batch_size=FLAGS.trim_batch_size,
        trim_grad_accum_steps=FLAGS.trim_grad_accum_steps,
        seed=FLAGS.seed,
        records_per_shard=FLAGS.records_per_shard,
        overwrite=FLAGS.overwrite,
    )
    print(out)


if __name__ == "__main__":
    app.run(main)
