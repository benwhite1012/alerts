import os
import requests
import json
from datetime import datetime
from requests.exceptions import HTTPError, Timeout, RequestException
import requests, urllib.parse, pathlib
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import asyncio, aiohttp, json, os, re, subprocess, tempfile, textwrap
import whisper, pathlib
import os, mimetypes, json
from openai import OpenAI
from typing import List, Optional
import logging
from dotenv import load_dotenv
import os, pathlib
import os, mimetypes, json
from openai import OpenAI     
import smtplib
from email.message import EmailMessage
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()

truth_social_user_id = os.environ["TS_ID"]

OPEN_AI_KEY = os.environ["OPENAI_API_KEY"]
SC_API_KEY = os.environ["SC_API_KEY"]

PROXY = os.environ["PROXY_URL"]

API_URL = os.environ["SC_POST_URL"]

EMAIL_USER = os.environ["EMAIL_USER"]
EMAIL_PASS = os.environ["EMAIL_PASS"]

TO_EMAIL = os.environ["TO_EMAIL"]
FROM_EMAIL = EMAIL_USER


HEADERS = {"x-api-key": SC_API_KEY}
BASE_PARAMS = {
    "user_id": truth_social_user_id,
    "trim": True
}

S3_BUCKET = os.environ["S3_BUCKET"]                # e.g. "my-alerts-bucket"
STATE_PREFIX = os.environ.get("STATE_PREFIX", "state/")  # optional "folder" in S3
ARCHIVE_KEY = f"{STATE_PREFIX}truth_post_archive.json"
LAST_POST_ID_KEY = f"{STATE_PREFIX}last_post_id.txt"
START_POST_ID_KEY = f"{STATE_PREFIX}start_post_id.txt"

s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))

def s3_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def s3_read_text(key: str) -> str:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read().decode("utf-8")

def s3_write_text(key: str, text: str) -> None:
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )

def s3_read_json(key: str):
    return json.loads(s3_read_text(key)) if s3_exists(key) else []

def s3_write_json(key: str, data) -> None:
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def get_last_seen_post_id():
    return s3_read_text(LAST_POST_ID_KEY).strip() if s3_exists(LAST_POST_ID_KEY) else None

def save_last_seen_post_id(post_id: str):
    s3_write_text(LAST_POST_ID_KEY, post_id)

def save_start_post_id(start_post_id: str | None):
    if start_post_id is not None:
        s3_write_text(START_POST_ID_KEY, start_post_id)
    else:
        # optional: clear file
        s3_write_text(START_POST_ID_KEY, "")

def get_start_post_id():
    return s3_read_text(START_POST_ID_KEY).strip() if s3_exists(START_POST_ID_KEY) else None

def load_saved_posts():
    return s3_read_json(ARCHIVE_KEY)

def save_archive(new_posts: list[dict]) -> int:
    """Read → merge-dedupe by id → write. Returns total count."""
    existing = s3_read_json(ARCHIVE_KEY)
    by_id = {p["id"]: p for p in existing}
    for p in new_posts:
        p["urls_intext"] = extract_links(p.get("text", ""))  # keep your enrichment
        by_id[p["id"]] = p
    combined = list(by_id.values())
    try:
        combined.sort(key=lambda p: int(p["id"]), reverse=True)
    except Exception:
        pass
    s3_write_json(ARCHIVE_KEY, combined)
    return len(combined)


def extract_links(text: str) -> list[str]:
    URL_RE = re.compile(r'https?://[^\s)\]\}>,]+')
    links = URL_RE.findall(text)

    # Only special-case: "RT: <link>" at the start
    if links and re.match(r'^\s*RT:\s*https?://', text, flags=re.I):
        first = links[0].rstrip(').,;]}>')  # trim common trailing punctuation

        # If the URL ends with DIGITS followed immediately by LETTERS, drop the letters
        m = re.search(r'(\d{5,})([A-Za-z]+)$', first)
        if m:
            links[0] = first[:m.end(1)]  # keep up to the end of the digit run

    return links

def fetch_new_trump_posts():
    last_seen_id = get_last_seen_post_id()
    start_post = last_seen_id
    save_start_post_id(start_post)
    new_posts = []
    next_max_id = None
    max_pages = 1 if last_seen_id is None else 1000
    page_count = 0

    while page_count < max_pages:
        page_count += 1
        params = BASE_PARAMS.copy()
        if next_max_id:
            params['next_max_id'] = next_max_id

        try:
            response = requests.get(API_URL, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                log.info(f"API call failed: {data}")
                break

            posts = data.get("posts", [])
            if not posts:
                break

            if last_seen_id is None:
                log.info("No last seen post found. Archiving first page of posts only.")
                new_posts.extend(posts)
                save_last_seen_post_id(posts[0]['id'])
                break

            for post in posts:
                if post['id'] == last_seen_id:
                    break
                new_posts.append(post)

            if any(post['id'] == last_seen_id for post in posts):
                break

            next_max_id = data.get("next_max_id")
            if not next_max_id:
                break

        except (HTTPError, Timeout, RequestException, ValueError) as err:
            log.exception(f"Request error: {err}")
            break

    #Save if any new posts
    if new_posts:
        log.info(f" {len(new_posts)} new post(s) found:\n")

        save_last_seen_post_id(new_posts[0]['id'])

        for post in new_posts:
          post['urls_intext'] = extract_links(post.get('text'))

        #Append to archive
        log.info("%d new post(s) found.", len(new_posts))
        save_last_seen_post_id(new_posts[0]['id'])
        total = save_archive(new_posts)
        log.info("Appended and saved %d new post(s). Archive now has %d items.", len(new_posts), total)
    else:
        log.info("No new posts since last check.")

def slice_from_id(posts: list[dict], start_id: str) -> list[dict]:
    if start_id is None or start_id == '': # Handle both None and empty string
        return posts
    try:
        start_id_int = int(start_id)
        return [p for p in posts if int(p.get("id", 0)) > start_id_int]
    except ValueError:
        # Handle cases where start_id is not a valid integer string
        log.exception(f"Warning: Invalid start_id '{start_id}'. Returning all posts.")
        return posts

def print_post_details(posts):
    for i, post in enumerate(posts, 1):
        log.info(f"Post #{i}")
        log.info(f"Created at: {post.get('created_at')}")
        log.info(f"ID: {post.get('id')}")
        log.info(f"URL: {post.get('url', '[No URL]')}")
        log.info(f"Text: {post.get('text', '[No text]')}")
        log.info(f"URLs in Text: {post.get('urls_intext', [])}")

        # Media
        media = post.get("media_attachments", [])
        if media:
            log.info("Media URLs:")
            for m in media:
                log.info(f"  - {m.get('url')}")

        # Embedded Link Card
        card = post.get("card")
        if card and card.get("url"):
            log.info("Embedded Link:")
            log.info(f"  - Title: {card.get('title')}")
            log.info(f"  - URL: {card.get('url')}")

        log.info("-" * 50)

def print_posts():
  start_post = get_start_post_id()
  log.info(f"Start post ID: {start_post}")
  posts = load_saved_posts()
  #choice = input("All posts (A) or new posts only (N)? ").strip().upper()
  #if choice == "A":
    #Run display
    #print_post_details(posts)
  #elif choice == "N":
    
  new_posts = slice_from_id(posts, start_post)
  if new_posts:
    print_post_details(new_posts)
  else:
      log.info("No new posts since last check.")
  #else:
    #log.info("Invalid input. Please enter 'A' or 'N'.")

def get_yt_id(url):
  m = re.search(r"(?:v=|youtu\.be/|embed/|live/|shorts/)([\w-]{11})", url)
  return m.group(1) if m else None

def yt_transcript(video_id, languages=["en"]):
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)   # ← new call
        # fetched behaves like a list of snippets; convert to plain text
        return " ".join(snippet["text"] for snippet in fetched.to_raw_data())
    except TranscriptsDisabled:
        log.exception("The channel disabled transcripts / CC.")
    except NoTranscriptFound:
        log.exception("No transcript available.")
    except Exception as e:
        log.exception("Other error:", exc_info=e)

WHISPER = whisper.load_model("base")

def transcribe_video(dest: pathlib.Path) -> str:
    result = WHISPER.transcribe(str(dest), language='en')
    text = result.get('text', '').strip()
    log.info("Transcribed %s (%.1f MB)", dest.name, dest.stat().st_size/1e6)
    return text

def get_proxied_url(url):
    return f"{PROXY}?u={urllib.parse.quote_plus(url)}"

def download_media(postid, url_ext):
    url, ext = url_ext
    stream_url = get_proxied_url(url)
    dest = pathlib.Path("/tmp") / f"{postid}.{ext.lstrip('.')}"
    with requests.get(stream_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    log.info("✓ downloaded %s (%.1f MB)", dest, dest.stat().st_size/1e6)
    return dest


ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

def allowed_images(media_list):
    """Return proxied URLs whose suffix is in ALLOWED_EXT."""
    return [
        get_proxied_url(m["url"])
        for m in media_list
        if pathlib.Path(m["url"]).suffix.lower() in ALLOWED_EXT
    ]

# === OpenAI set-up ============================================================

client = OpenAI(                     # make a client instance
    api_key=OPEN_AI_KEY
)

SYSTEM_PROMPT = """
You are a risk-scanner for Donald Trump’s Truth Social posts.

TASK
────
1. Read the user message (may include text and/or one image).
2. Decide whether the post could influence:
     • the overall economy
     • financial markets (equities, bonds, FX, crypto, commodities)
     • monetary policy, jobs, or trade
3. Classify the likely market impact:
     • MAJOR  – probable to move broad markets or a major sector
     • MINOR  – directional but small or sector-specific
     • NONE – choose only when label is "NO" (step 2 is false)
4. Produce the JSON object described below—nothing else.

OUTPUT FORMAT
─────────────
Return **only** valid JSON with exactly these keys:

{
  "label": "YES" | "NO",                         # YES if step 2 true
  "impact": "MAJOR" | "MINOR" | "NONE",          # see step 3
  "reason": "<≤15 words>",                       # terse logic for logs
  "brief_post_summary": "<≤100 words>",     # 2-sentence recap
  "economical_impacts": "<≤100 words>"      # 2-sentence of potential economical impacts
  "headline": "<≤10 words>"                      # if impact==MAJOR, prefix and suffix with emojis and sensationalism, keep as short as possible
}
"""

def classify_post(*, text: Optional[str] = None, image_urls: Optional[List[str]] = None) -> dict:

    user_content = []
    if text:
        user_content.append({"type": "text", "text": text[:6000]})
    if image_urls:
        for url in image_urls:
            user_content.append({"type": "image_url", "image_url": {"url": url}})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",                 # or "gpt-4o"
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0
    )
    return json.loads(resp.choices[0].message.content)

# --- CONSTANT ----------------------------------------------------------------
SEARCH_PROMPT = (
    "Search recent, reputable news (<30 days). "
    "Return ONE paragraph (≤120 words) focused on market-moving angles. "
    "Include at least one trailing citation."
)

# --- STEP B: enrich only when label == YES  ----------------------------------
def enrich_if_relevant(result: dict) -> dict:
    """Add a 'news_summary' paragraph via web_search_preview when label is YES."""
    if result.get("label") != "YES":
        return result          # nothing to do

    seed = result["brief_post_summary"][:200]

    # ---- Responses API call with the search tool ----------------------------
    rsp = client.responses.create(
        model="gpt-4o-mini",                      # cheap & quick
        tools=[{ "type": "web_search_preview" }],
        tool_choice={ "type": "web_search_preview" },   # force one search
        input=seed,
        instructions=SEARCH_PROMPT,               # same text you used before
        temperature=0.3
    )

    # The summarised paragraph is in rsp.output_text
    result["news_summary"] = rsp.output_text.strip()
    return result

# --- PIPELINE ----------------------------------------------------------------
def analyse_post(text: str | None = None, image_urls=None) -> dict:
    base_json = classify_post(text=text, image_urls=image_urls)  # your existing fn
    return enrich_if_relevant(base_json)

def remove_chat_refferal(result):
  summary = result.get("news_summary")
  if summary:
    result["news_summary"] = summary.replace("?utm_source=openai", "")
  return result

def send_email_alert(subject, body, to_email):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email

    # Send using Gmail SMTP
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)

def trigger_alert(post, result):
    text, url = post

    post_summary = result.get('brief_post_summary')
    economical_impact = result['economical_impacts']
    news_summary = result.get('news_summary')

    subject = result['headline']

    body = f"Trump just posted: {text}\n\nFind the post at: {url}\n\nPost Summary: {post_summary}\n\nPossible Economical Impacts: {economical_impact}"
    if news_summary:
        body += f"\n\nRelevant News Summary: {news_summary}"

    # Send alerts
    send_email_alert(subject, body, TO_EMAIL)

# ========= MAIN PIPELINE =====================================================
def run():
    """
    One-shot orchestrator:
      • fetch -> save -> process -> alert
    """
    # -------------------------------------------------------------------------
    # A.  SCRAPE LATEST POSTS & UPDATE THE ARCHIVE
    # -------------------------------------------------------------------------
    fetch_new_trump_posts()                      # already saves to JSON + last id
    # -------------------------------------------------------------------------
    # B.  LOAD ONLY THE POSTS WE HAVEN’T ANALYSED THIS SESSION
    # -------------------------------------------------------------------------
    start_id  = get_start_post_id()              # the id grabbed just before fetch
    all_posts = load_saved_posts()               # whole archive
    new_posts = slice_from_id(all_posts, start_id)

    if not new_posts:
        log.info("✓ No fresh posts to handle.")
        return

    log.info(f"✓ {len(new_posts)} fresh post(s) queued for analysis.\n")

    print_posts()

    # -------------------------------------------------------------------------
    # C.  PROCESS EACH POST
    # -------------------------------------------------------------------------
    email_count = 0
    for p in new_posts:
        try:
            text          = p.get("text", "")
            in_text_links = p.get("urls_intext", [])

            # --- 2. fetch / generate extra transcript data -------------------
            transcript_chunks = []

            # 2a. YouTube links found in body
            for link in in_text_links:
                vid_id = get_yt_id(link)
                if vid_id:
                    yt_txt = yt_transcript(vid_id) or ""
                    transcript_chunks.append(yt_txt)

            # 2b. Any attached MP4 videos
            for m in p.get("media_attachments", []):
                if (m.get("type") == "video") and m.get("url", "").endswith(".mp4"):
                    dest = None
                    try:
                        dest = download_media(p["id"], (m["url"], ".mp4"))
                        transcript_chunks.append(transcribe_video(dest))
                    except Exception as e:
                        log.exception("Video transcript failed for post %s: %s", p.get("id"), e)
                    finally:
                        if dest is not None:
                            try:
                                dest.unlink(missing_ok=True)
                            except Exception as e:
                                log.warning("Could not delete temp file %s: %s", dest, e)

            full_text = (text + "\n\n" + "\n\n".join(transcript_chunks)).strip()

            # --- 3. push through the AI pipeline -----------------------------
            # Use first image (if any) for classification context
            image_urls = allowed_images(p.get("media_attachments", []))
            result  = analyse_post(text=full_text, image_urls=image_urls)
            log.info(f"{json.dumps(result, indent=2)}")

            # --- 4. alert when appropriate -----------------------------------
            if result["label"].lower() == "yes":
                result  = remove_chat_refferal(result)
                post_url = p.get("url")
                trigger_alert((text, post_url), result)
                email_count += 1
                log.info(f"→ Alert sent for post {p['id']}")
            else:
                log.info(f"→ Post {p['id']} classified as NO impact")

        except Exception as e:
            log.exception(f"Error handling post {p.get('id')}: {e}")

    log.info(f"\n✓ Run complete. {len(new_posts)} posts were retrieved and {email_count} email notifications were sent")

# ---------------------------------------------------------------------------
# OPTIONAL COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()