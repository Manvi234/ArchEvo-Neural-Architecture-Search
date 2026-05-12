#!/usr/bin/env python3
"""
run_recommender.py
------------------
Train the ArchEvo recommender on existing results and run inference
on a new dataset to suggest the best architecture + pressure mode.

Usage:
    python run_recommender.py \
        --dataset_name cifar10 \
        --results_db results/experiments.db \
        --output_dir results/recommender/

    # With a custom data directory:
    python run_recommender.py \
        --data_dir data/my_dataset \
        --dataset_name eurosat \
        --results_db results/experiments.db \
        --output_dir results/recommender/
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('run_recommender')


def parse_args():
    p = argparse.ArgumentParser(
        description="ArchEvo Architecture Recommender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dataset_name', type=str, required=True,
                   help='Dataset name to profile and recommend for '
                        '(known: cifar10, eurosat, isic, cub200; '
                        'or provide --data_dir for unknown datasets)')
    p.add_argument('--data_dir', type=str, default=None,
                   help='Path to dataset root directory (ImageFolder structure). '
                        'If provided, data will be profiled from this directory.')
    p.add_argument('--results_db', type=str, default='results/experiments.db',
                   help='Path to SQLite experiments database')
    p.add_argument('--output_dir', type=str, default='results/recommender',
                   help='Directory to save recommender state and recommendations')
    p.add_argument('--recommender_path', type=str, default=None,
                   help='Load pre-fitted recommender from this path '
                        '(skip fitting if provided)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--max_profile_batches', type=int, default=50,
                   help='Max batches to process when profiling dataset')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from archevo.recommender import ArchitectureRecommender, DatasetProfiler

    # -------------------------------------------------------------------
    # 1. Initialise recommender
    # -------------------------------------------------------------------
    recommender = ArchitectureRecommender(results_db_path=args.results_db)

    if args.recommender_path and os.path.exists(args.recommender_path):
        logger.info(f"Loading pre-fitted recommender from: {args.recommender_path}")
        recommender.load(args.recommender_path)
    else:
        logger.info("Fitting recommender on existing experiments...")
        recommender.fit()
        logger.info(recommender.summary())

        # Save fitted recommender
        recommender_save_path = os.path.join(args.output_dir, 'recommender.pkl')
        recommender.save(recommender_save_path)
        logger.info(f"Recommender saved to: {recommender_save_path}")

    # -------------------------------------------------------------------
    # 2. Profile dataset (if data_dir provided)
    # -------------------------------------------------------------------
    dataset_loader = None

    if args.data_dir is not None:
        logger.info(f"Profiling dataset from: {args.data_dir}")
        try:
            import torchvision.transforms as T
            from torchvision.datasets import ImageFolder
            from torch.utils.data import DataLoader

            transform = T.Compose([
                T.Resize(64),
                T.CenterCrop(64),
                T.ToTensor(),
            ])

            # Try train split, then root directly
            train_dir = os.path.join(args.data_dir, 'train')
            if os.path.isdir(train_dir):
                ds = ImageFolder(root=train_dir, transform=transform)
            elif os.path.isdir(args.data_dir):
                ds = ImageFolder(root=args.data_dir, transform=transform)
            else:
                raise RuntimeError(f"Could not find dataset at: {args.data_dir}")

            dataset_loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            logger.info(f"Dataset loaded: {len(ds)} images, {len(ds.classes)} classes")
        except Exception as e:
            logger.warning(f"Could not load dataset from {args.data_dir}: {e}")
            logger.info("Falling back to known dataset features...")
            dataset_loader = None

    # -------------------------------------------------------------------
    # 3. Get recommendation
    # -------------------------------------------------------------------
    logger.info(f"Generating recommendation for dataset: {args.dataset_name}")

    if dataset_loader is not None:
        profiler = DatasetProfiler(max_batches=args.max_profile_batches)
        stats = profiler.compute_stats(dataset_loader)
        logger.info("Dataset statistics:")
        for k, v in stats.items():
            if isinstance(v, list):
                formatted = [f"{x:.4f}" for x in v] if all(isinstance(x, float) for x in v) else v
                logger.info(f"  {k}: {formatted}")
            else:
                logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        recommendation = recommender.recommend(
            dataset_loader=dataset_loader,
            dataset_name=args.dataset_name,
        )
    else:
        recommendation = recommender.recommend(
            dataset_name=args.dataset_name,
        )

    # -------------------------------------------------------------------
    # 4. Display and save recommendation
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Architecture Recommendation")
    print("=" * 60)
    print(f"  Dataset:            {args.dataset_name}")
    print(f"  Matched to:         {recommendation.get('matched_dataset', 'N/A')}")
    print(f"  Expected accuracy:  {recommendation.get('expected_accuracy', 0):.4f}")
    print(f"  Recommended pressure mode: {recommendation.get('pressure_mode', 'none')}")

    nearest_dist = recommendation.get('nearest_distance', None)
    if nearest_dist is not None:
        print(f"  Nearest-neighbor distance: {nearest_dist:.4f}")

    genotype = recommendation.get('genotype', [])
    print(f"\nRecommended Genotype:")
    if genotype:
        for node_idx, node_ops in enumerate(genotype):
            print(f"  Node {node_idx}: {node_ops}")
    else:
        print("  (no genotype found — run more experiments)")

    print("=" * 60)

    # Save recommendation to file
    rec_path = os.path.join(args.output_dir, f'recommendation_{args.dataset_name}.json')
    with open(rec_path, 'w') as f:
        # JSON-serialise (convert numpy types if any)
        def _serialise(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if hasattr(obj, 'item'):
                return obj.item()
            return obj

        serialisable = {
            k: _serialise(v) for k, v in recommendation.items()
            if k != 'stats'  # stats may contain tensors
        }
        if 'stats' in recommendation:
            stats_s = {}
            for k, v in recommendation['stats'].items():
                if isinstance(v, list):
                    stats_s[k] = [float(x) for x in v]
                elif isinstance(v, (int, float)):
                    stats_s[k] = v
                else:
                    stats_s[k] = str(v)
            serialisable['stats'] = stats_s

        json.dump(serialisable, f, indent=2)

    logger.info(f"Recommendation saved to: {rec_path}")

    # Suggest next steps
    print("\nNext steps:")
    if genotype:
        genotype_path = os.path.join(args.output_dir, f'recommended_genotype_{args.dataset_name}.json')
        with open(genotype_path, 'w') as f:
            json.dump(genotype, f, indent=2)
        print(f"  1. Evaluate the recommended architecture:")
        print(f"     python run_eval.py \\")
        print(f"       --genotype_path {genotype_path} \\")
        print(f"       --dataset {args.dataset_name} \\")
        print(f"       --epochs 100")
    else:
        print(f"  1. Run architecture search first:")
        print(f"     python run_search.py --dataset {args.dataset_name} --algorithm darts")

    pressure = recommendation.get('pressure_mode', 'none')
    if pressure and pressure != 'none':
        print(f"  2. Apply recommended pressure '{pressure}':")
        print(f"     python run_search.py --dataset {args.dataset_name} --pressure {pressure}")


if __name__ == '__main__':
    main()
