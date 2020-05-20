# Import packages
import os
import pandas as pd

# Define constants
BASE_DIR = 'data'
MOVIELENS_DIR = BASE_DIR + '/ml-100k/'
USER_DATA_FILE = 'users.dat'
MOVIE_DATA_FILE = 'u.item'
RATING_DATA_FILE = 'u.data'

AGES = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATIONS = {0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
               4: "college/grad student", 5: "customer service", 6: "doctor/health care",
               7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
               12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
               17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}

RATING_CSV_FILE = 'ml100k_ratings.csv'
USER_CSV_FILE = 'ml100k_users.csv'
MOVIE_CSV_FILE = 'ml100k_movies.csv'

ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE),
                      sep='\t',
                      engine='python',
                      encoding='latin-1',
                      names=['userid', 'movieid', 'rating', 'timestamp'])

max_userid = ratings['userid'].drop_duplicates().max()
max_movieid = ratings['movieid'].drop_duplicates().max()
# Begin indexing at zero
ratings['user_emb_id'] = ratings['userid'] - 1
ratings['movie_emb_id'] = ratings['movieid'] - 1
print(len(ratings), 'ratings loaded')

ratings.to_csv(RATING_CSV_FILE,
               sep='\t',
               header=True,
               encoding='latin-1',
               columns=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])
print('Saved to', RATING_CSV_FILE)

movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE),
                     sep='\t',
                     engine='python',
                     encoding='latin-1',
                     names=['movieid', 'title', 'genre'])
print(len(movies), 'descriptions of', max_movieid, 'movies loaded')
movies['movie_emb_id'] = movies['movieid'] - 1
movies.to_csv(MOVIE_CSV_FILE,
              sep='\t',
              header=True,
              columns=['movie_emb_id', 'title', 'genre'])
print('Saved to', MOVIE_CSV_FILE)

print(len(ratings['userid'].drop_duplicates()), 'of the', max_userid, 'users rate at least one movie.')
print(len(movies['movieid'].drop_duplicates()), 'of the', max_movieid, 'movies are rated.')
