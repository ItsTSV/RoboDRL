import argparse as ap


def get_training_args():
    """Parse all training arguments."""
    parser = ap.ArgumentParser(
        description="Reinforcement Learning Trainer"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="What config file to use?"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="online",
        help="Will the training be logged? [online, disabled]"
    )

    args = parser.parse_args()
    if args.config is None:
        raise ValueError("Please, provide a config file using --config argument.")

    return args


def get_sb3_args():
    """Parse all StableBaselines3 arguments."""
    parser = ap.ArgumentParser(
        description="StableBaselines3 Benchmarker"
    )
    parser.add_argument(
        "--env",
        type=str,
        help="What environment to train on?"
    )
    parser.add_argument(
        "--alg",
        type=str,
        help="What algorithm to use?"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
        help="How many training steps?"
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=100,
        help="How many episodes will be used for evaluation?"
    )

    args = parser.parse_args()
    if args.env is None or args.alg is None:
        raise ValueError("Please, provide a valid environment and algorithm using --env and --alg switches.")

    return args


def get_hf_args():
    """Get attributes for HuggingFace mass upload."""
    parser = ap.ArgumentParser(
        description="HuggingFace upload manager"
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Username that will be used to create HF repositories"
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="*.yaml",
        help="What models will be uploaded?"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="URL of HuggingFace collection"
    )
    parser.add_argument(
        "--skip-existing",
        type=bool,
        default=True,
        help="Whether to skip models for which repositories already exist"
    )

    args = parser.parse_args()
    if args.username is None:
        raise ValueError("Please, provide your username with --username <str> switch")

    return args
