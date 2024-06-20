from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import aiohttp
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from torchvision import transforms
import asyncio
from geoguessr import Streetview

device = 'cuda'
model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(device)
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

cities_df = pd.read_csv('./1k.csv', delimiter=';')
tile_url = "https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"

app = FastAPI()
origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictPayload(BaseModel):
    panoID: str
    
    
def is_black(pixel):
    return pixel == (0, 0, 0) or pixel == (0, 0, 0, 255)

def find_non_black_bounds(image):
    width, height = image.size
    pixels = image.load()

    left = 0
    while left < width and all(is_black(pixels[left, y]) for y in range(height)):
        left += 1

    right = width - 1
    while right > left and all(is_black(pixels[right, y]) for y in range(height)):
        right -= 1

    top = 0
    while top < height and all(is_black(pixels[x, top]) for x in range(width)):
        top += 1

    bottom = height - 1
    while bottom > top and all(is_black(pixels[x, bottom]) for x in range(width)):
        bottom -= 1

    return left, top, right + 1, bottom + 1

def crop_black_bars(image):
    bbox = find_non_black_bounds(image)
    if bbox != (0, 0, image.width, image.height):
        cropped_image = image.crop(bbox)
        return cropped_image
    else:
        return image
    

def update_csv(cities_df):
    cities_df.to_csv("worldcities.csv", index=False)
    
async def download_tile(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.read() 
        else:
            print(f"Failed to download tile from {url}")
            return None

async def download_and_stitch_tiles(pano_id, zoom=3, tile_size=512, i=0):
    num_horizontal_tiles = 1 << zoom
    num_vertical_tiles = num_horizontal_tiles // 2

    panorama_image = Image.new('RGB', (num_horizontal_tiles * tile_size, num_vertical_tiles * tile_size))

    max_x, max_y = -1, -1

    async with aiohttp.ClientSession() as session:
        tasks = [
            (x, y, download_tile(session, tile_url.format(pano_id=pano_id, zoom=zoom, x=x, y=y)))
            for x in range(num_horizontal_tiles)
            for y in range(num_vertical_tiles)
        ]

        results = await asyncio.gather(*(task[2] for task in tasks))

        for (x, y), tile_data in zip(((x, y) for x in range(num_horizontal_tiles) for y in range(num_vertical_tiles)), results):
            if tile_data:
                tile = Image.open(BytesIO(tile_data))
                panorama_image.paste(tile, (x * tile_size, y * tile_size))
                max_x, max_y = max(max_x, x), max(max_y, y)

    if max_x >= 0 and max_y >= 0:
        cropped_image = panorama_image.crop((0, 0, (max_x + 1) * tile_size, (max_y + 1) * tile_size))
        return cropped_image
    else:
        print(f"Failed to create panorama for {pano_id}, no valid tiles found.")
        return None
    

@app.post("/api/v1/predict")
async def predict(payload: PredictPayload):
    global cities_df
    pano = payload.panoID
    image = await download_and_stitch_tiles(pano_id=pano,zoom=3)
    image = crop_black_bars(image)
    image.save(f"./{pano}.jpg")
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(1.3)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    transform(enhanced_image).unsqueeze(0).to(device)

    # Predict the country
    country_names = list(Streetview.abbrvs.keys())
    country_inputs = processor(text=country_names, images=[enhanced_image], return_tensors="pt", padding=True)
    country_inputs = {k: v.to(device) for k, v in country_inputs.items()}
    country_outputs = model(**country_inputs)
    country_probs = country_outputs.logits_per_image.softmax(dim=1).tolist()[0]
    country_prob_dict = {country_names[i]: country_probs[i] for i in range(len(country_probs))}


    predicted_country = max(country_prob_dict, key=country_prob_dict.get)
    country_code = Streetview.abbrvs.get(predicted_country, None)

    country_df = cities_df[cities_df['Country Code'] == country_code]
    filtered_cities_df = country_df[country_df['Population'] > 1500]
    cities_df = cities_df[~cities_df['Geoname ID'].isin(country_df[~country_df['ASCII Name'].isin(filtered_cities_df['ASCII Name'])]['Geoname ID'])]

    city_names = filtered_cities_df['ASCII Name'].dropna().tolist()
    batch_size = 64 
    city_probs = []

    # Predict the town / city
    for i in range(0, len(city_names), batch_size):
        batch_city_names = city_names[i:i+batch_size]

        if not all(isinstance(city, str) for city in batch_city_names):
            continue
        
        city_inputs = processor(text=batch_city_names, images=[enhanced_image], return_tensors="pt", padding=True)
        city_inputs = {k: v.to(device) for k, v in city_inputs.items()}
        city_outputs = model(**city_inputs)
        city_probs.extend(city_outputs.logits_per_image.softmax(dim=1).tolist()[0])

    city_prob_dict = {city_names[i]: city_probs[i] for i in range(len(city_probs))}

    predicted_city = max(city_prob_dict, key=city_prob_dict.get)

    city_data = filtered_cities_df[filtered_cities_df['ASCII Name'] == predicted_city].iloc[0]
    lat, lon = map(float, city_data['Coordinates'].split(','))
    
    return {"lat": lat, "lng":lon}
