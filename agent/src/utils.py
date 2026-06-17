# © Copyright European Union - 2026

import json

def extract_json_from_reply(raw_output) -> str:

	""" determine which part of the output contains a valid JSON

	only dict and list are supposed, assumes the related special character do not appears elsewhere than in the JSON"""

	try:
		json.loads(raw_output)
		return raw_output
	except:
		pass

	tentative_json = None
	try:
		raw_output = raw_output.strip()
		if "```" in raw_output:
			raw_output = raw_output[raw_output.index("```"):raw_output.rindex("```")]
		if raw_output.startswith("```json") and raw_output.endswith("```"):
			raw_output = raw_output.lstrip("```json").rstrip("```")
		idx_start = None
		if "{" in raw_output and "}" in raw_output:
			idx_start = raw_output.index("{")
			idx_end = raw_output.index("}")
		if "[" in raw_output and "]" in raw_output:
			if ((idx_start is not None) and (idx_start > raw_output.index("["))) or idx_start is None:
				idx_start = raw_output.index("[")
				idx_end = raw_output.rindex("]")

		if idx_start is None:
			tentative_json = raw_output
		else:
			tentative_json = raw_output[idx_start:idx_end+1]	
		json.loads(tentative_json)
		return tentative_json
	except Exception as e:
		print(e)
		raise Exception("ERROR JSON impossible to extract", raw_output, tentative_json)