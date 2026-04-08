def main():
    import argparse
    from chacra.windowed_frequencies import (
        compute_windowed_frequencies,
        percentile_windowed_frequencies,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compute contact frequencies from a windowed subset of frames.\n"
            "Slices per-frame contact files via Polars and passes the result\n"
            "to get-contact-frequencies. Fast and memory-efficient."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--contacts_dir", "-c", type=str, required=True,
        help="Directory containing per-frame contact files.",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True,
        help=(
            "Output directory for frequency files. "
            "When --percentiles is used, subdirectories are created here."
        ),
    )
    parser.add_argument(
        "--file_pattern", "-f", type=str, default="cont_state_{state}.tsv",
        metavar="PATTERN",
        help=(
            "Filename pattern with {state} placeholder. Use * for multi-run "
            "(e.g. 'run_*/contacts/cont_state_{state}.tsv'). "
            "Default: cont_state_{state}.tsv"
        ),
    )
    parser.add_argument(
        "--n_states", "-n", type=int, default=None,
        help="Number of states. Auto-discovered if not specified.",
    )
    parser.add_argument(
        "--end_frame", type=int, default=None,
        help="Inclusive end frame index (for single-window mode).",
    )
    parser.add_argument(
        "--percentiles", type=float, nargs="+", default=None,
        metavar="PCT",
        help=(
            "Compute frequencies at these trajectory percentile cutoffs "
            "(e.g. 50 60 70 80 90 100). Creates one subdirectory per "
            "percentile under --output_dir."
        ),
    )
    parser.add_argument(
        "--reference_state", type=int, default=0,
        help="State used to determine total frame count (default: 0).",
    )
    parser.add_argument(
        "--n_jobs", "-j", type=int, default=1,
        help="Number of states to process in parallel (default: 1).",
    )

    args = parser.parse_args()

    if args.percentiles is not None:
        results = percentile_windowed_frequencies(
            contacts_dir=args.contacts_dir,
            base_output_dir=args.output_dir,
            percentiles=args.percentiles,
            file_pattern=args.file_pattern,
            n_states=args.n_states,
            reference_state=args.reference_state,
            n_jobs=args.n_jobs,
        )
    elif args.end_frame is not None:
        paths = compute_windowed_frequencies(
            contacts_dir=args.contacts_dir,
            output_dir=args.output_dir,
            end_frame=args.end_frame,
            file_pattern=args.file_pattern,
            n_states=args.n_states,
            n_jobs=args.n_jobs,
        )
        print(f"\nDone. Wrote {len(paths)} frequency files to {args.output_dir}/")
    else:
        parser.error("Specify either --percentiles or --end_frame.")


if __name__ == "__main__":
    main()
