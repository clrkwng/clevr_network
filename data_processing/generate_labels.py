"""
This file parses a CLEVR_scenes.json file generated for CLEVR
dataset images and gets the number of cubes, cylinders, spheres.
This serves as the label for each image: (num_cubes, num_cylinders, num_spheres).
"""

import json

# Returns a tuple of (num_cubes, num_cylinders, num_spheres) from json_path.
def parse_objects_from_json(json_path):
	with open(json_path) as f:
		data = json.load(f)

	num_cubes = 0
	num_cylinders = 0
	num_spheres = 0

	for o in data["scenes"][0]["objects"]:
		shape_name = o["shape"]
		num_cubes += 1 if shape_name == "cube" else 0
		num_cylinders += 1 if shape_name == "cylinder" else 0
		num_spheres += 1 if shape_name == "sphere" else 0

	return (num_cubes, num_cylinders, num_spheres)