import json, sys


def read_json(filename):
	with open(filename) as f:
		data = json.load(f)
	return data

def get_job_specification(config_file):
	json_data = read_json(config_file)
	return json_data