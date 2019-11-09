import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

client_id= "1c152cf60bb94f8695836484148e1d4b"
client_secret= "5ce0cdc1c9594c23b857919f8a7e4bd9"

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#Utility functions
def show_tracks(tracks):
    for i, item in enumerate(tracks['items']):
        track = item['track']
        print ("   %32.32s %s" % (track['artists'][0]['name'], track['name']))
def show_playlist(username, playlist_id):
    results = sp.user_playlist(username, playlist_id, fields="tracks,next")
    tracks = results['tracks']
    while tracks:
        show_tracks(tracks)
        tracks = sp.next(tracks)
def get_playlist_tracks(username, playlist_id):
    return_value = []
    results = sp.user_playlist(username, playlist_id, fields="tracks,next")
    tracks = results['tracks']
    while tracks:
        return_value += [ item['track'] for (i, item) in enumerate(tracks['items']) ]
        tracks = sp.next(tracks)
    return return_value
def get_playlist_URIs(username, playlist_id):
    return [t["uri"] for t in get_playlist_tracks(username, playlist_id)]
def splitlist(l,n) :
    t = l[:]
    r = []
    while len(t) :
        r += [t[0:n]]
        t = t[n:]
    return r
def get_audio_features (track_URIs) :
    features = []
    for pack in splitlist(track_URIs,50) :
        features += sp.audio_features(pack)
    df = pd.DataFrame.from_dict(features)
    df["uri"] = track_URIs
    return df


#Collecting playlist URI
#Playlist "Piano Classical" by user yguezennec
#Playlist "Rock" by user yguezennec
username = "yguezennec"
playlists = sp.user_playlists(username)

while playlists:
    for i, playlist in enumerate(playlists['items']):
        print("%4d %s %s = %s tracks" % (i + 1 + playlists['offset'], playlist['uri'],  playlist['name'],playlist['tracks']['total']))
    if playlists['next']:
        playlists = sp.next(playlists)
    else:
        playlists = None

#URI of Piano Classical Playlist is "183mAeiSAAUZm40gvvB1he" 344 tracks
#URI of Rock Playlist is "0NY0QaRzStMKbHHuGWpG1K" 120 tracks

classic_URIs = get_playlist_URIs(username, "183mAeiSAAUZm40gvvB1he")
rock_URIs = get_playlist_URIs(username, "0NY0QaRzStMKbHHuGWpG1K")

#Get features in playlists
class_feat = get_audio_features(classic_URIs) #344 rows x 18 columns
rock_feat = get_audio_features(rock_URIs) #120 rows x 18 columns

#Tagging
class_feat["playlists"] = "piano_classical"
rock_feat["playlists"] = "rock"

#Concatinating
df_all = pd.concat([class_feat, rock_feat], sort=True, ignore_index=True) #464 rows, 19 columns

df_all.columns
#returns: ['acousticness', 'analysis_url', 'danceability', 'duration_ms', 'energy',
       # 'id', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       # 'playlists', 'speechiness', 'tempo', 'time_signature', 'track_href',
       # 'type', 'uri', 'valence']

#Features to analyse: acousticness, danceability, energy, instrumentalness,
#liveness, loudness, speechiness, tempo

#Creating new data frame
df = df_all[['acousticness', 'danceability', 'energy','instrumentalness',
            'liveness', 'loudness','speechiness','tempo','uri','playlists']]

#Exporting data into a csv
# df.to_csv (r'\export_dataframe.csv', header=True)
