"""
fetch_data.py

Fetch the full AniList anime dataset from the AniList GraphQL API and save it
as flat tabular files for downstream analysis and modeling.

This script:
- queries AniList anime records through the GraphQL API
- works around large-result limits by fetching data in year-based batches
- flattens nested API fields into dataframe-friendly columns
- preserves complex fields (for example tags, relations, staff, and stats)
  as JSON strings
- saves the final dataset to CSV, Excel, and Pickle formats

Outputs:
- data/raw/anilist_anime_data_complete.csv
- data/raw/anilist_anime_data_complete.xlsx
- data/raw/anilist_anime_data_complete.pkl

Notes:
- Uses FuzzyDateInt year ranges to page through the full anime catalog
- Stores intermediate yearly batch files in temp_anime_data/
- Supports a test mode for smaller trial runs
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('fetch_data')

ANILIST_API = "https://graphql.anilist.co"

QUERY = """
query ($page: Int, $perPage: Int, $startDate: FuzzyDateInt, $endDate: FuzzyDateInt) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
      perPage
    }
    media(type: ANIME, startDate_greater: $startDate, startDate_lesser: $endDate) {

      id
      idMal
      title {
        romaji
        english
        native
        userPreferred
      }
      type
      format
      status
      description
      startDate {
        year
        month
        day
      }
      endDate {
        year
        month
        day
      }
      season
      seasonYear
      seasonInt
      episodes
      duration
      chapters
      volumes
      countryOfOrigin
      isLicensed
      source
      hashtag
      trailer {
        id
        site
        thumbnail
      }
      updatedAt
      coverImage {
        extraLarge
        large
        medium
        color
      }
      bannerImage
      

      genres
      synonyms
      tags {
        id
        name
        description
        category
        rank
        isGeneralSpoiler
        isMediaSpoiler
        isAdult
      }
      

      averageScore
      meanScore
      popularity
      favourites
      trending
      rankings {
        id
        rank
        type
        format
        year
        season
        allTime
        context
      }
      

      isFavourite
      isAdult
      isLocked
      

      siteUrl
      externalLinks {
        id
        url
        site
        type
        language
        color
        icon
        notes
        isDisabled
      }
      streamingEpisodes {
        title
        thumbnail
        url
        site
      }
      

      relations {
        edges {
          id
          relationType
          node {
            id
            title {
              romaji
              english
              native
            }
            type
            format
            status
          }
        }
      }
      

      characters {
        edges {
          id
          role
          name
          voiceActors {
            id
            name {
              full
              native
            }
            languageV2
            image {
              large
              medium
            }
          }
          node {
            id
            name {
              full
              native
              alternative
            }
            image {
              large
              medium
            }
            description
          }
        }
      }
      

      staff {
        edges {
          id
          role
          node {
            id
            name {
              full
              native
            }
            languageV2
            image {
              large
              medium
            }
          }
        }
      }
      

      studios {
        edges {
          id
          isMain
          node {
            id
            name
            isAnimationStudio
          }
        }
      }
      

      nextAiringEpisode {
        id
        airingAt
        timeUntilAiring
        episode
        mediaId
      }
      airingSchedule {
        nodes {
          id
          airingAt
          timeUntilAiring
          episode
          mediaId
        }
      }
      

      recommendations {
        edges {
          node {
            id
            rating
            mediaRecommendation {
              id
              title {
                romaji
                english
                native
              }
            }
          }
        }
      }
      

      reviews {
        edges {
          node {
            id
            summary
            rating
            score
          }
        }
      }
      

      stats {
        scoreDistribution {
          score
          amount
        }
        statusDistribution {
          status
          amount
        }
      }
    }
  }
}
"""

def convert_to_fuzzy_date(year, month=1, day=1):
    """
    Convert year, month, day to FuzzyDateInt format required by AniList API
    
    FuzzyDateInt format: YYYYMMDD as integer
    
    Args:
        year (int): Year
        month (int, optional): Month (1-12). Defaults to 1.
        day (int, optional): Day (1-31). Defaults to 1.
        
    Returns:
        int: Date in FuzzyDateInt format
    """
    return year * 10000 + month * 100 + day

def fetch_anime_page(page, per_page=50, start_year=None, end_year=None):
    """
    Fetch a single page of anime data from AniList GraphQL API
    
    Args:
        page (int): Page number to fetch
        per_page (int): Number of items per page
        start_year (int): Start year for filtering (inclusive)
        end_year (int): End year for filtering (inclusive)
        
    Returns:
        dict: JSON response from AniList API
    """

    start_date = convert_to_fuzzy_date(start_year, 1, 1) if start_year else None
    end_date = convert_to_fuzzy_date(end_year, 12, 31) if end_year else None
    
    variables = {
        'page': page,
        'perPage': per_page,
        'startDate': start_date,
        'endDate': end_date
    }
    
    payload = {
        'query': QUERY,
        'variables': variables
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    
    try:
        response = requests.post(ANILIST_API, json=payload, headers=headers)
        

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limited. Waiting for {retry_after} seconds...")
            time.sleep(retry_after)
            return fetch_anime_page(page, per_page, start_year, end_year)
        
        if response.status_code != 200:
            logger.error(f"Error: {response.status_code}")
            logger.error(response.text)
            return None
        
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching anime page: {str(e)}")
        return None

def flatten_anime_data(anime):
    """
    Flatten nested anime data into a dictionary suitable for pandas DataFrame
    
    Args:
        anime (dict): Anime data from AniList API
        
    Returns:
        dict: Flattened anime data
    """
    flattened = {}
    
    flattened['id'] = anime.get('id')
    flattened['idMal'] = anime.get('idMal')
    
    title = anime.get('title', {})
    flattened['title_romaji'] = title.get('romaji')
    flattened['title_english'] = title.get('english')
    flattened['title_native'] = title.get('native')
    flattened['title_userPreferred'] = title.get('userPreferred')
    
    flattened['type'] = anime.get('type')
    flattened['format'] = anime.get('format')
    flattened['status'] = anime.get('status')
    flattened['description'] = anime.get('description')
    
    start_date = anime.get('startDate', {})
    flattened['startDate_year'] = start_date.get('year')
    flattened['startDate_month'] = start_date.get('month')
    flattened['startDate_day'] = start_date.get('day')
    
    end_date = anime.get('endDate', {})
    flattened['endDate_year'] = end_date.get('year')
    flattened['endDate_month'] = end_date.get('month')
    flattened['endDate_day'] = end_date.get('day')
    
    flattened['season'] = anime.get('season')
    flattened['seasonYear'] = anime.get('seasonYear')
    flattened['seasonInt'] = anime.get('seasonInt')
    
    flattened['episodes'] = anime.get('episodes')
    flattened['duration'] = anime.get('duration')
    flattened['chapters'] = anime.get('chapters')
    flattened['volumes'] = anime.get('volumes')
    
    flattened['countryOfOrigin'] = anime.get('countryOfOrigin')
    flattened['isLicensed'] = anime.get('isLicensed')
    flattened['source'] = anime.get('source')
    flattened['hashtag'] = anime.get('hashtag')
    
    trailer = anime.get('trailer') or {}
    flattened['trailer_id'] = trailer.get('id')
    flattened['trailer_site'] = trailer.get('site')
    flattened['trailer_thumbnail'] = trailer.get('thumbnail')
    
    flattened['updatedAt'] = anime.get('updatedAt')
    
    cover_image = anime.get('coverImage', {})
    flattened['coverImage_extraLarge'] = cover_image.get('extraLarge')
    flattened['coverImage_large'] = cover_image.get('large')
    flattened['coverImage_medium'] = cover_image.get('medium')
    flattened['coverImage_color'] = cover_image.get('color')
    flattened['bannerImage'] = anime.get('bannerImage')
    
    flattened['genres'] = json.dumps(anime.get('genres', []))
    flattened['synonyms'] = json.dumps(anime.get('synonyms', []))
    
    tags = anime.get('tags', [])
    flattened['tags'] = json.dumps(tags)
    
    flattened['averageScore'] = anime.get('averageScore')
    flattened['meanScore'] = anime.get('meanScore')
    flattened['popularity'] = anime.get('popularity')
    flattened['favourites'] = anime.get('favourites')
    flattened['trending'] = anime.get('trending')
    
    rankings = anime.get('rankings', [])
    flattened['rankings'] = json.dumps(rankings)
    
    flattened['isFavourite'] = anime.get('isFavourite')
    flattened['isAdult'] = anime.get('isAdult')
    flattened['isLocked'] = anime.get('isLocked')
    
    flattened['siteUrl'] = anime.get('siteUrl')
    
    external_links = anime.get('externalLinks', [])
    flattened['externalLinks'] = json.dumps(external_links)
    
    streaming_episodes = anime.get('streamingEpisodes', [])
    flattened['streamingEpisodes'] = json.dumps(streaming_episodes)
    
    relations = anime.get('relations', {}).get('edges', [])
    flattened['relations'] = json.dumps(relations)
    
    characters = anime.get('characters', {}).get('edges', [])
    flattened['characters'] = json.dumps(characters)
    
    staff = anime.get('staff', {}).get('edges', [])
    flattened['staff'] = json.dumps(staff)
    
    studios = anime.get('studios', {}).get('edges', [])
    flattened['studios'] = json.dumps(studios)
    
    next_airing_episode = anime.get('nextAiringEpisode', {})
    flattened['nextAiringEpisode'] = json.dumps(next_airing_episode) if next_airing_episode else None
    
    airing_schedule = anime.get('airingSchedule', {}).get('nodes', [])
    flattened['airingSchedule'] = json.dumps(airing_schedule)
    
    recommendations = anime.get('recommendations', {}).get('edges', [])
    flattened['recommendations'] = json.dumps(recommendations)
    
    reviews = anime.get('reviews', {}).get('edges', [])
    flattened['reviews'] = json.dumps(reviews)
    
    stats = anime.get('stats', {})
    score_distribution = stats.get('scoreDistribution', [])
    status_distribution = stats.get('statusDistribution', [])
    flattened['stats_scoreDistribution'] = json.dumps(score_distribution)
    flattened['stats_statusDistribution'] = json.dumps(status_distribution)
    
    return flattened

def fetch_all_anime(test_mode=False):
    """
    Fetch all anime from AniList API using year-based filtering to overcome the 5,000 item limitation
    
    Args:
        test_mode (bool): Whether to run in test mode (limited data)
        
    Returns:
        pandas.DataFrame: DataFrame containing all anime data
    """
    if test_mode:
        year_ranges = [(2020, 2020)]
        logger.info("Running in TEST MODE - only fetching anime from 2020")
    else:
        year_ranges = [
            (1940, 1965),
            (1966, 1970), (1971, 1975), (1976, 1980),
            (1981, 1985), (1986, 1990), (1991, 1995), (1996, 2000),
            (2001, 2005), (2006, 2007), (2008, 2009),
            (2010, 2011), (2012, 2013), (2014, 2015),
            (2016, 2016), (2017, 2017), (2018, 2018),
            (2019, 2019), (2020, 2020), (2021, 2021),
            (2022, 2022), (2023, 2023), (2024, 2024),
            (2025, 2025),(2026,2026)
        ]
    
    all_anime = []
    

    temp_dir = Path("temp_anime_data")
    temp_dir.mkdir(exist_ok=True)
    

    for start_year, end_year in year_ranges:
        logger.info(f"Fetching anime from {start_year} to {end_year}...")
        
        anime_batch = []
        page = 1
        has_next_page = True
        

        with tqdm(desc=f"{start_year}-{end_year}", unit="page") as pbar:
            while has_next_page:

                response = fetch_anime_page(page, per_page=50, start_year=start_year, end_year=end_year)
                
                if not response or 'data' not in response:
                    logger.error(f"Failed to fetch page {page} for years {start_year}-{end_year}")
                    break
                

                page_info = response['data']['Page']['pageInfo']
                media_list = response['data']['Page']['media']
                

                for anime in media_list:

                    flattened_anime = flatten_anime_data(anime)
                    anime_batch.append(flattened_anime)
                

                pbar.update(1)
                pbar.set_postfix({"anime": len(anime_batch), "page": page})
                

                has_next_page = page_info['hasNextPage']
                page += 1
                

                if test_mode and page > 2:
                    logger.info("Test mode: stopping after 2 pages")
                    break
                

                time.sleep(0.5)
        
        if anime_batch:

            batch_df = pd.DataFrame(anime_batch)
            temp_file = os.path.join(temp_dir, f"anime_{start_year}_{end_year}.pkl")
            batch_df.to_pickle(temp_file)
            logger.info(f"Saved {len(anime_batch)} anime to {temp_file}")
            
            all_anime.extend(anime_batch)
    

    df = pd.DataFrame(all_anime)
    

    if not df.empty:
        df = df.drop_duplicates(subset=['id'])
        logger.info(f"After removing duplicates: {len(df)} unique anime entries")
    
    return df

def main():
    """Main function to fetch all anime and save to CSV"""

    parser = argparse.ArgumentParser(description='AniList Anime Data Scraper')
    parser.add_argument('--test', action='store_true', help='Run in test mode (fetch only a few pages)')
    args = parser.parse_args()
    
    logger.info("Starting AniList anime data scraper...")
    

    df = fetch_all_anime(test_mode=args.test)
    
    if df.empty:
        logger.error("Failed to fetch anime data")
        return
      

    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {raw_dir}")
    else:
        logger.info(f"Directory already exists: {raw_dir}")
        

    csv_filename = raw_dir / "anilist_anime_data_complete.csv"
    df.to_csv(csv_filename, index=True)
    logger.info(f"Saved {len(df)} anime records to {csv_filename}")
    

    try:
        excel_filename = raw_dir / "anilist_anime_data_complete.xlsx"
        df.to_excel(excel_filename, index=True)
        logger.info(f"Saved {len(df)} anime records to {excel_filename}")
    except Exception as e:
        logger.error(f"Warning: Could not save to Excel format: {e}")
    

    pickle_filename = raw_dir / "anilist_anime_data_complete.pkl"
    df.to_pickle(pickle_filename)
    logger.info(f"Saved {len(df)} anime records to {pickle_filename}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
