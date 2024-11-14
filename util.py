import json
from shapely import Polygon
from codes import CODE2, CODE3, months
import random
import requests
from time import sleep
import torch
from transformers import CLIPModel
from models import CustomCLIPWrapper
from backbones_utils import load_backbone
import glob
import os

def custom2clip(model_dir, backbone):
    clip_model, _ = load_backbone(backbone)
    clip_keys = list(clip_model.state_dict().keys())
    print(f'{len(clip_keys)} parameters in model')
    
    ckpt_files = glob.glob(f'{model_dir}/*.ckpt')
    
    for custom_ckpt in ckpt_files:
        print(f'Converting {custom_ckpt}...')
        convert_name = custom_ckpt.split('ckpt')[0][:-1] + '_clip.pth'
        if not os.path.exists(convert_name):
            # Load custom model parameters
            ckpt = torch.load(custom_ckpt, map_location='cpu')
            custom_state_dict = ckpt['state_dict']
            custom_keys = list(custom_state_dict.keys())

            # Convert custom to clip
            convert_state_dict = {}

            i = 0
            for k in clip_keys:
                parts = k.split('.')
                if len(parts) <= 2:
                    i+=1
                    custom_name = [name for name in custom_keys if parts[0] in name][0]
                elif parts[0] == 'text_model':
                    i+=1
                    custom_name = 'model.transformer.'+'.'.join(parts[1:])
                elif parts[0] == 'vision_model':
                    i+=1
                    custom_name = 'model.visual.'+'.'.join(parts[1:])
                v = custom_state_dict[custom_name]
                convert_state_dict[k] = v
            print(f'{i} parameters converted')

            # Check compatibility
            ret = clip_model.load_state_dict(convert_state_dict)
            all_keys_matched = len(ret.missing_keys) == 0 and len(ret.unexpected_keys) == 0

            # If successful, save new state dict
            if ret:
                torch.save(convert_state_dict, convert_name)
                print(f'Saved converted model to {convert_name}')
            print()
        else:
            print('skipping')
    

months = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

def get_meta_from_json(file_dir, geolocator=None):
    
    with open(file_dir, 'r') as file:
        data = json.load(file)
    
    polygon = data['raw_location'][10:-2]
    coords = convert_poly(polygon)
    poly = Polygon(coords)
    lon, lat = poly.centroid.x, poly.centroid.y
    country_code = data['country_code']
    country = CODE3.get(country_code, CODE2.get(country_code, country_code))
    if geolocator is not None:
        city, state, country = get_loc_geopy(lat, lon, geolocator)
    else:
        city, state, country = '', '', ''
    
    metadata = {
        'img_dir': file_dir.replace('_gt', '').replace('json', 'jpg'),
        'label': file_dir.split('/')[-3].replace('_', ' '),
        'lat': lat,
        'lon': lon,
        'city': city,
        'state': state,
        'country': country,
        'country_code': country_code,
        'country_from_code': country,
        'gsd': data['gsd'],
        'cloud_cover': data['cloud_cover'],
        'year': int(data['timestamp'][:4]),
        'month': months[int(data['timestamp'][5:7])],
        'day': int(data['timestamp'][8:10])
    }
    
    ### Remove caption here (generate different versions with dataloader)
    #caption = create_fmow_caption(metadata) 
    #metadata['caption'] = caption

    return metadata

def get_meta_from_json_mb(file_dir, access_token):
    with open(file_dir, 'r') as file:
        data = json.load(file)
    
    polygon = data['raw_location'][10:-2]
    coords = convert_poly(polygon)
    poly = Polygon(coords)
    lon, lat = poly.centroid.x, poly.centroid.y
    country_code = data['country_code']
    country_from_code = CODE3.get(country_code, CODE2.get(country_code, country_code))
    
    # Construct the API URL
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json?access_token={access_token}"

    # Make the request
    response = requests.get(url)

    # Check the status code and parse the response
    if response.status_code == 200:
        geodata = response.json()
        if len(geodata['features']) > 0:
            place_info = geodata['features'][0]
            # Extracting city, state, and country
            city = None
            state = None
            country = None

            # The place_info 'context' will include detailed breakdown
            if 'context' in place_info:
                for feature in place_info['context']:
                    if 'place' in feature['id']:
                        city = feature['text']
                    if 'region' in feature['id']:
                        state = feature['text']
                    if 'country' in feature['id']:
                        country = feature['text']
            else:
                print("No place information found.")
        else:
            print("No place information found.")
    else:
        print(f"Failed to retrieve data: {response.status_code}")
    
    sleep(0.05)
    
    metadata = {
        'img_dir': file_dir.replace('_gt', '').replace('json', 'jpg'),
        'label': file_dir.split('/')[-3].replace('_', ' '),
        'lat': lat,
        'lon': lon,
        'city': city,
        'state': state,
        'country': country,
        'country_code': country_code,
        'country_from_code': country,
        'gsd': data['gsd'],
        'cloud_cover': data['cloud_cover'],
        'year': int(data['timestamp'][:4]),
        'month': months[int(data['timestamp'][5:7])],
        'day': int(data['timestamp'][8:10])
    }

    return metadata

def create_fmow_caption(metadata, caption_type=0, drop_pct=0):
    
    str_incl = lambda x: x if random.random() > drop_pct else ''
    
    cls_name = metadata['label']
    gsd = metadata['gsd']
    lon, lat = metadata['lon'], metadata['lat']
    year, month, day = metadata['year'], metadata['month'], metadata['day']
    city = '' if isinstance(metadata['city'], float) else metadata['city']
    state = '' if isinstance(metadata['state'], float) else metadata['state']
    country = metadata['country']
    
    comma_1 = ',' if city else ''
    comma_2 = ',' if state else ''
    
    if caption_type == 0:
        caption = (f"A satellite image"
                   f"{str_incl(f' of a {cls_name}')}"
                   f"{str_incl(f' in {city}{comma_1} {state}{comma_2} {country}.')}"
                  )
    elif caption_type == 1:
            caption = (f"A satellite image"
               f"{str_incl(f' of a {cls_name}')}"
               f"{str_incl(f' in {city}{comma_1} {state}{comma_2} {country}.')}"
               f"{str_incl(f' The date is {month} {day}, {year}.')}"
              )
    elif caption_type == 2:
            caption = (f"A satellite image"
               f"{str_incl(f' of a {cls_name}')}"
               f"{str_incl(f' in {city}{comma_1} {state}{comma_2} {country}.')}"
               f"{str_incl(f' The date is {month} {day}, {year}.')}"
               f"{str_incl(f' The ground sample distance is {round(gsd, 5)} meters.')}"
               #f"{str_incl(f' The longitude, latitude is {lon:.3f}, {lat:.3f}.')}"
              )
    
    return caption

def get_coords(data):
    polygon = data['raw_location'][10:-2]
    coords = convert_poly(polygon)
    poly = Polygon(coords)
    lon, lat = poly.centroid.x, poly.centroid.y
    return lon, lat

def convert_poly(poly_str):
    components = poly_str.replace(',', '').split()
    tuples = [(float(components[i]), float(components[i + 1])) for i in range(0, len(components), 2)]
    return tuples

def get_loc_geopy(lat, lon, geolocator):
    
    location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
    address = location.raw['address']
    city = address.get('city', '')
    state = address.get('state', '')
    country = address.get('country', '')

    return city, state, country
