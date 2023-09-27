from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi


def get_youtube_video_transcript(url: str) -> str:
    parse_result = urlparse(url)
    dict_result = parse_qs(parse_result.query)

    video_id = dict_result["v"][0]

    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    text = " ".join([d["text"] for d in transcript]).strip()

    return text


url = "https://www.youtube.com/watch?v=nZromH6F7R0&list=PLeo1K3hjS3uuvuAXhYjV2lMEShq2UYSwX&index=19"

text = get_youtube_video_transcript(url)

word_count = len(text.split())
