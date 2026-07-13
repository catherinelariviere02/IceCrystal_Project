import argparse

import signac
from equilibrate import equilibrate
from Analysis import analyze
from initialize import initialize 

if __name__ == "__main__":
    #parse command line arguments: python project.py --action <ACTION> [DIRECTORIES]
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True)
    parser.add_argument("directories", nargs="+")
    args = parser.parse_args()

    #open signac jobs
    project = signac.get_project("../../data/workspace")
    jobs = [project.open_job(id=directory) for directory in args.directories]

    if args.action == "initialize":
        initialize(*jobs)
    if args.action == "equilibrate":
        equilibrate(*jobs)
    elif args.action == "analyze":
        analyze(*jobs)

