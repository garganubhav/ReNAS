#!/usr/bin/python3

import sys
from utils import get_job_specification
from stubs.pytorch.schedule_job import schedule_job


def main():
	if len(sys.argv) == 2:
		config_file = sys.argv[1]
	else:
		config_file = 'configs/experiment-2.json'
	job_specification = get_job_specification(config_file)
	schedule_job(job_specification)


if __name__ == '__main__':
	main()
