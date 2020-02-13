import json 
from os import walk, path, makedirs

from shapely import wkt
from shapely.geometry import Polygon
import numpy as np 
from cv2 import fillPoly, imwrite


def get_files(base_dir):
    files = []
    
    dis_pre_files = [f for f in next(walk(path.join(base_dir, "labels")))[2] if 'post' in f]
    for f in dis_pre_files:
        files.append(path.join(base_dir, "labels", f))

    return files


def create_image(inference_data):
    # This is the same function from inference_image_output.py with slight modifications
    damage_key = {'un-classified': 0, 'no-damage': 1, 'minor-damage': 2, 'major-damage': 3, 'destroyed': 4}

    mask_img = np.zeros((1024,1024,1), np.uint8)
    
    for poly in inference_data['features']['xy']:
        if 'subtype' in poly['properties']:
            damage = poly['properties']['subtype']
        else:
            # If the subtype json field does not exist, do not write out the polygon 
            damage = 'un-classified'

        coords = wkt.loads(poly['wkt'])
        poly_np = np.array(coords.exterior.coords, np.int32)
        
        fillPoly(mask_img, [poly_np], damage_key[damage])
    
    return mask_img


def save_image(polygons, output_path):
    # This is the same function from inference_image_output.py
    # Output the filled in polygons to an image file
    imwrite(output_path, polygons)

def write_gt(infile, output_dir):
    with open(infile) as gt_file:
        gt_json = json.load(gt_file)

        # Localization gt
        gt_masked_image = create_image(gt_json)
        gt_masked_image_path = path.join(output_dir, path.basename(infile).split('.json')[0]+'_masked_dmg.png')
        save_image(gt_masked_image, gt_masked_image_path)

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="perfect_scores.py: produces perfect scores for the train set")

    parser.add_argument('--base-dir',
                        required=True,
                        metavar='/path/to/xBD/train/',
                        help="Full path to the train directory to produce perfect scores for testing")
    parser.add_argument('--output-dir',
                        required=True,
                        metavar='/path/to/output/directory/',
                        help="Full path to the train directory to produce perfect scores for testing")

    args = parser.parse_args()

    # Create output dir to save all masks if it doesn't exist already
    if not path.isdir(args.output_dir):
        makedirs(args.output_dir)

    # We expect all label files to be under a base dir like:
    # ~/Downloads/train/labels/<ALL_LABELS>.png
    all_files = get_files(args.base_dir)

    for infile in all_files:
        write_gt(infile, args.output_dir)
