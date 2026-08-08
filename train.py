"""
train.py

Entry point: trains the CNN on normalized-image .npz files, checkpointing
often enough that a crash never loses more than a few steps of progress.

USAGE
-----
    python train.py --data /path/to/normalized_npz_dir --ckpt checkpoints/run1

Re-running the exact same command after a crash (or a deliberate
Ctrl-C) automatically resumes from the last checkpoint — same shard,
same sample index, same optimizer momentum, same weights.

WHAT "checkpointing" MEANS HERE
--------------------------------
- Every `--save_every` steps (default 50), and once at the very end of
  each shard file, the FULL state is atomically written to
  <ckpt>.npz / <ckpt>.json (see checkpoint.py): every weight, AdamW's
  m/v moment estimates, its global step counter, and exactly which
  shard/sample to resume from.
- If train_step() raises partway through a sample (bad image, NaN,
  out-of-memory, you kill -9 the process, etc.), the `finally` block
  still saves an emergency checkpoint from the last *successfully
  completed* step before the exception propagates and the process
  exits — so you only ever lose the one in-flight sample, never
  everything trained so far.
- Optimizer state (m/v) is what actually makes resuming safe for
  AdamW specifically: reloading only the weights and restarting m/v/t
  from zero would give the first several post-resume steps much
  larger effective updates than intended.

PERFORMANCE NOTE (read this before pointing it at the real dataset)
---------------------------------------------------------------------
conv1.py/conv2.py/conv3.py run their forward AND backward convolution
loops as explicit Python `for` loops over every output spatial
position (out_h * out_w iterations, each doing a small matmul). At the
real pipeline resolution (720x720 -> ... -> 354x354) that is roughly
500k Python-level loop iterations for conv1 alone, per image, for
forward AND again for backward. In pure Python/NumPy (no JIT, no
vectorized im2col) this will be SLOW — plausibly minutes per image
depending on your machine, which multiplied across a real RDD2022
India split (thousands of images) could be many hours to days for one
epoch. The smoke tests in this project use tiny synthetic images
(24x24) specifically to prove the wiring is correct in seconds; they
say nothing about how long a real-resolution run will take. If that
turns out to be impractical, the fix is to vectorize conv1/conv2/conv3
forward+backward with an im2col + single matmul instead of the
double-for-loop — happy to do that as a follow-up if you hit this.
"""

from __future__ import annotations

import argparse
import sys
import time

from checkpoint import load_trainflow, save_checkpoint
from data_loader import iter_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train the road-damage CNN on normalized npz images.")
    p.add_argument("--data", required=True, help="Directory of normalized-image .npz shard files.")
    p.add_argument("--ckpt", required=True, help="Checkpoint path prefix, e.g. checkpoints/run1")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--save_every", type=int, default=50, help="Checkpoint every N training steps.")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--dropout_rate", type=float, default=0.40)
    p.add_argument(
        "--labels_are_datapipeline_order",
        action="store_true",
        help="Set this if your npz labels are integers written using datapipeline.py's "
             "(Normal=0,Crack=1,Pothole=2,Both=3) order instead of the model's own "
             "(Pothole=0,Crack=1,Both=2,Normal=3) order. String labels never need this flag.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    tf, progress = load_trainflow(
        args.ckpt,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
    )

    start_epoch = progress.get("epoch", 0)
    start_shard = progress.get("shard_index", 0)
    start_sample = progress.get("sample_index", 0)
    running_loss = progress.get("running_loss") or 0.0
    steps_done = tf.optimizer.t

    print(f"Resuming from step {steps_done} "
          f"(epoch {start_epoch}, shard {start_shard}, sample {start_sample})"
          if steps_done > 0 else "Starting fresh training run.")

    step_in_run = 0
    t0 = time.time()

    try:
        for epoch in range(start_epoch, args.epochs):
            shard_start = start_shard if epoch == start_epoch else 0
            sample_start = start_sample if epoch == start_epoch else 0

            for shard_index, sample_index, image, target_class in iter_dataset(
                args.data,
                labels_are_datapipeline_order=args.labels_are_datapipeline_order,
                start_shard_index=shard_start,
                start_sample_index=sample_start,
            ):
                result = tf.train_step(image, target_class, dropout_seed=tf.optimizer.t)
                step_in_run += 1
                running_loss = 0.98 * running_loss + 0.02 * result["loss"] if steps_done > 0 else result["loss"]
                steps_done += 1

                progress = {
                    "epoch": epoch,
                    "shard_index": shard_index,
                    # resume AFTER this sample next time
                    "sample_index": sample_index + 1,
                    "running_loss": running_loss,
                }

                if step_in_run % 10 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"epoch {epoch} shard {shard_index} sample {sample_index} "
                        f"| step {steps_done} | loss {result['loss']:.4f} "
                        f"(running {running_loss:.4f}) | acc {result['accuracy']:.0f} "
                        f"| {elapsed:.1f}s elapsed"
                    )

                if step_in_run % args.save_every == 0:
                    save_checkpoint(args.ckpt, tf, progress=progress)
                    print(f"  [checkpoint saved at step {steps_done}]")

            # end of shard loop for this epoch -> next epoch starts at shard 0, sample 0
            progress = {"epoch": epoch + 1, "shard_index": 0, "sample_index": 0, "running_loss": running_loss}
            save_checkpoint(args.ckpt, tf, progress=progress)
            print(f"[checkpoint saved at end of epoch {epoch}]")

        print("Training complete.")

    except KeyboardInterrupt:
        print("\nInterrupted by user — saving emergency checkpoint before exiting...")
        raise

    except Exception:
        print("\nTraining crashed — saving emergency checkpoint from the last completed step...", file=sys.stderr)
        raise

    finally:
        # Always persist the last successfully completed step, whatever happened.
        save_checkpoint(args.ckpt, tf, progress=progress)
        print(f"Checkpoint saved to {args.ckpt}.npz / {args.ckpt}.json")


if __name__ == "__main__":
    main()
