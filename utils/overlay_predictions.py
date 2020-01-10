import json 
from PIL import Image, ImageDraw
from IPython.display import display
from shapely import wkt

def save_img(path_to_image, path_to_label, path_to_output): 
    with open(path_to_label, 'rb') as image_json_file:
        image_json = json.load(image_json_file)
    
    coords = image_json['features']['xy']
    wkt_polygons = []
    
    for coord in coords:
        if 'subtype' in coord['properties']:
            damage = coord['properties']['subtype']
        else:
            damage = 'no-damage'
        wkt_polygons.append((damage, coord['wkt']))
        
    polygons = []
    
    for damage, swkt in wkt_polygons:
        polygons.append((damage, wkt.loads(swkt)))
    
    # Loading image
    img = Image.open(path_to_image) 
    
    draw = ImageDraw.Draw(img, 'RGBA')
    
    damage_dict = {
        "no-damage": (0, 255, 0, 100),
        "minor-damage": (0, 0, 255, 125),
        "major-damage": (255, 69, 0, 125),
        "destroyed": (255, 0, 0, 125),
        "un-classified": (255, 255, 255, 125)
    }
    
    for damage, polygon in polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict[damage])

    img.save(path_to_output)

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """overlay_predictions.py: writes predictions over the image and saves the image with the polygons overtop"""
    )
    parser.add_argument('--image',
                        required=True,
                        metavar='/path/to/input/image.png',
                        help="Full path to the image to use to overlay the predictions onto")
    parser.add_argument('--json',
                        required=True,
                        metavar='/path/to/polygons/and/classifications.json',
                        help="Full path to the prediction json output from model"
    )
    parser.add_argument('--output',
                        required=True,
                        metavar='/path/to/save/img_pred.png',
                        help="Full path to save the final single output file to (include filename.png)"
    )

    args = parser.parse_args()

    # Combining the json based off the uuid assigned at the polygonize stage
    save_img(args.image, args.json, args.output)
