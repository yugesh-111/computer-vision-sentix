
from googleapiclient.discovery import build,HttpError

api_key = 'AIzaSyB_d69XgLRxeoN__MfEMF93vi6ZqwQkJxo'

# Initialize the YouTube Data API
from urllib.parse import urlparse, parse_qs

def extract_video_id(url): 
    parsed_url = urlparse(url)
    
    if parsed_url.netloc in ('www.youtube.com', 'youtu.be'):
        query_params = parse_qs(parsed_url.query)
        if 'v' in query_params:
            video_id = query_params['v'][0]
            return video_id
        elif parsed_url.path.startswith('/embed/') or parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[-1]
        elif parsed_url.netloc == 'youtu.be':
            return parsed_url.path.lstrip('/')
    
    return None
youtube = build('youtube', 'v3', developerKey=api_key)

# Specify the video ID
# url='https://www.youtube.com/watch?v=_KvtVk8Gk1A'
# video_id = extract_video_id(url)
# https://youtu.be/7UVoCmolAPI

# Retrieve comments from the video
def extract_comment(video_id):
    comments = []

    try:
        nextPageToken = None
        while True:
            if len(comments) >= 500:
                break
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=100,
                pageToken=nextPageToken
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                break

        # Save the comments to a file
        with open('comments.txt', 'w', encoding='utf-8') as f:
            for comment in comments:
                f.write(comment + '\n')

        print(f'Successfully scraped {len(comments)} comments and saved to "comments.txt".')
    except HttpError as e:
        error_message = e.content.decode('utf-8')
        if 'commentsDisabled' in error_message:
            print("Comments are disabled for this video.")
        else:
            print(f"An error occurred while fetching comments: {error_message}")
    
    return comments






