
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import openai
from openai import OpenAI
import time

#openai.api_key = "API Key here"

locations = [
    "New York, USA",
    "Los Angeles, USA",
    "Chicago, USA",
    "Houston, USA",
    "Phoenix, USA",
    "Philadelphia, USA",
    "San Antonio, USA",
    "San Diego, USA",
    "Dallas, USA",
    "San Jose, USA",
    "London, UK",
    "Birmingham, UK",
    "Glasgow, UK",
    "Manchester, UK",
    "Liverpool, UK",
    "Bristol, UK",
    "Toronto, Canada",
    "Montreal, Canada",
    "Vancouver, Canada",
    "Calgary, Canada",
    "Ottawa, Canada",
    "Edmonton, Canada",
    "Paris, France",
    "Marseille, France",
    "Lyon, France",
    "Toulouse, France",
    "Nice, France",
    "Nantes, France",
    "Berlin, Germany",
    "Hamburg, Germany",
    "Munich, Germany",
    "Cologne, Germany",
    "Frankfurt, Germany",
    "Stuttgart, Germany",
    "Tokyo, Japan",
    "Osaka, Japan",
    "Nagoya, Japan",
    "Sapporo, Japan",
    "Fukuoka, Japan",
    "Kyoto, Japan",
    "Sydney, Australia",
    "Melbourne, Australia",
    "Brisbane, Australia",
    "Perth, Australia",
    "Adelaide, Australia",
    "Cairo, Egypt",
    "Alexandria, Egypt",
    "Nairobi, Kenya",
    "Cape Town, South Africa",
    "Johannesburg, South Africa",
    "Moscow, Russia",
    "Saint Petersburg, Russia",
    "Dubai, UAE",
    "Abu Dhabi, UAE",
    "Mumbai, India",
    "Delhi, India",
    "Bangalore, India",
    "Chennai, India",
    "Shanghai, China",
    "Beijing, China",
    "Guangzhou, China",
    "Chongqing, China",
    "Hong Kong, China",
    "Seoul, South Korea",
    "Busan, South Korea",
    "Istanbul, Turkey",
    "Ankara, Turkey",
    "Buenos Aires, Argentina",
    "Sao Paulo, Brazil",
    "Rio de Janeiro, Brazil",
    "Lima, Peru",
    "Mexico City, Mexico",
    "Santiago, Chile",
    "Madrid, Spain",
    "Barcelona, Spain",
    "Rome, Italy",
    "Milan, Italy",
    "Athens, Greece",
    "Stockholm, Sweden",
    "Oslo, Norway",
    "Helsinki, Finland",
    "Dublin, Ireland",
    "Vienna, Austria",
    "Zurich, Switzerland",
    "Lisbon, Portugal"
]


client = OpenAI(api_key = openai.api_key)
model = "gpt-4o-mini"
columns = [
    "_unit_id", "gender", "created", "description", "fav_number",
    "name", "retweet_count", "text", "tweet_count",
    "tweet_created", "tweet_id", "tweet_location", "user_timezone"
]

genders = ["male", "female", "brand"]

def generate_tweet(gender, location):
    prompt = (
        f"You are a {gender} based in {location}. Write a tweet."
        "The tweet does not have to include explicit references to gender or location."
        "Only use alphanumeric characters, no emojis."
    )
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.4,
            max_tokens=50,
            stop=["\n"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating tweet: {e}")
        return None

def generate_bio(gender, location):
    prompt = (
        f"You are a {gender} from {location}. Write a Twitter bio."
        "The bio does not have to explictly state gender or location."
        "Only use alphanumeric characters, no emojis."
    )
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=30,
            stop=["\n"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating bio: {e}")
        return None

def generate_username(gender, location):
    prompt = (
        f"Generate a Twitter username for a {gender} in {location}. "
        "The username does not have to explictly state gender or location."
        "Do no include anything in the output except for the username."
    )
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=15,
            stop=["\n"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating username: {e}")
        return None

tweet_rows = []

num_tweets = 2000
for i in range(num_tweets):
    print(f"Generating tweet {i}")
    gender = random.choice(genders)
    location = random.choice(locations)

    tweet_text = generate_tweet(gender, location)
    bio = generate_bio(gender, location)
    username = generate_username(gender, location)

    if tweet_text is None or bio is None or username is None:
        print(f"Warning: Failed to generate data for tweet {i}. Skipping.")
        continue

    row = [
        i,               # unit_id
        gender,          # gender
        pd.NaT,          # created
        bio,             # description
        pd.NA,           # fav_number
        username,        # name
        pd.NA,           # retweet_count
        tweet_text,      # text
        pd.NA,           # tweet_count
        pd.NaT,          # tweet_created
        pd.NA,           # tweet_id
        location,        # tweet_location
        pd.NA           # user_timezone
    ]

    tweet_rows.append(row)
    time.sleep(1)

df = pd.DataFrame(tweet_rows, columns=columns)
df.to_csv("generated_tweets_2000.csv", index=False)
print(f"Saved {num_tweets} generated tweets to generated_tweets.csv")


df.head()


