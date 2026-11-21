#!/usr/bin/env python3
"""
Torrent & Downloaded Media Organizer
====================================
Organizes downloaded movies and web series into a clean, IMDB-style folder hierarchy
categorized by original production language:

Target Directory Structure:
---------------------------
- Movies:
    <target_dir>/Movies/<Language>/<Movie Name> (<Year>)/<Movie Name> (<Year>).<ext>
    (including subtitles renamed to match: <Movie Name> (<Year>).srt)

- Web Series:
    <target_dir>/Web Series/<Language>/<Series Name> (<1st Season Year>)/<Series Name> S<Season2Digits>/<Series Name> S<Season2Digits>E<Episode2Digits>[ <Episode Name>].<ext>
    Example: Kurukshetra (2025) -> Kurukshetra S01 -> [Kurukshetra S01E01.mp4, ...]
    (including subtitles renamed to match: <Series Name> S<Season2Digits>E<Episode2Digits>.srt)

Key Features:
-------------
- 1 single prompt / LLM call (via local OpenAI-compatible LM Studio endpoint) to extract content type, clean title, year, season, and episode.
- Direct Wikipedia MediaWiki API & IMDb suggestion API integration for canonical metadata.
- Primary production language from metadata, with a conservative fallback to one unambiguous release-name language tag.
- Handles nested subdirectories, flat episode layouts, already structured folders, and junk file filtering.
- After an actual run, removes source files other than non-sample videos and subtitles; dry-run previews cleanup.
- Colored logging with distinct levels and icons.
- CLI arguments for source_dir, target_dir, --dry-run, --generate-csv-only, --parallelism, --cleanup, and connection options.
"""

import os
import sys
import re
import json
import shutil
import logging
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm.auto import tqdm
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

try:
    import openai
except ImportError:
    openai = None

# ==============================================================================
# Console & Logging Setup
# ==============================================================================
console = Console()

def setup_logging(level: str = "INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)]
    )

logger = logging.getLogger("media_organizer")

# ==============================================================================
# File Extension Definitions
# ==============================================================================
VIDEO_EXTENSIONS = {'.mkv', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.m4v', '.webm', '.ts'}
SUBTITLE_EXTENSIONS = {'.srt', '.sub', '.vtt', '.idx', '.ass', '.ssa'}
IGNORE_EXTENSIONS = {
    '.nfo', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.url',
    '.torrent', '.exe', '.htm', '.html', '.ds_store', '.part',
    '.aria2', '.xml', '.nfo-orig', '.db'
}

# Standard language mapping & normalization
LANGUAGE_ALIASES = {
    'telgu': 'Telugu',
    'telugu': 'Telugu',
    'kannad': 'Kannada',
    'kannada': 'Kannada',
    'tamil': 'Tamil',
    'tam': 'Tamil',
    'malayalam': 'Malayalam',
    'mal': 'Malayalam',
    'hindi': 'Hindi',
    'hin': 'Hindi',
    'english': 'English',
    'eng': 'English',
    'tel': 'Telugu',
    'kan': 'Kannada',
    'mar': 'Marathi',
    'ben': 'Bengali',
    'pun': 'Punjabi',
    'guj': 'Gujarati',
    'ita': 'Italian',
    'jpn': 'Japanese',
    'filipino': 'Tagalog',
    'tagalog': 'Tagalog',
    'korean': 'Korean',
    'kor': 'Korean',
    'japanese': 'Japanese',
    'jap': 'Japanese',
    'spanish': 'Spanish',
    'french': 'French',
    'german': 'German',
    'italian': 'Italian',
    'chinese': 'Chinese',
    'mandarin': 'Mandarin',
    'cantonese': 'Cantonese',
    'bengali': 'Bengali',
    'marathi': 'Marathi',
    'punjabi': 'Punjabi',
    'gujarati': 'Gujarati',
    'russian': 'Russian',
    'thai': 'Thai',
    'turkish': 'Turkish',
    'swedish': 'Swedish',
    'danish': 'Danish',
    'norwegian': 'Norwegian',
}

def normalize_language_name(lang_str: Optional[str]) -> str:
    """Normalizes raw language string to clean title-cased language name."""
    if not lang_str:
        return 'Unknown'
    # Clean text
    clean = re.sub(r'\(.*?\)', '', str(lang_str)).strip()
    words = re.findall(r'[a-zA-Z]+', clean)
    for word in words:
        w_lower = word.lower()
        if w_lower in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[w_lower]
    # Return first word capitalized if alphabetic
    if words:
        return words[0].capitalize()
    return 'Unknown'

def extract_release_year(name: str) -> Optional[int]:
    """Return the last plausible year in a release name (usually the release year)."""
    years = re.findall(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)', str(name))
    for candidate in reversed(years):
        year = valid_year(candidate)
        if year:
            return year
    return None

_RELEASE_MARKER_RE = re.compile(
    r'\b(?:480|576|720|1080|1440|2160)\s*p\b|\b4k\b|\b8k\b|'
    r'\b(?:WEB[ ._-]?(?:DL|RIP|MUX)|HDTV|HDRIP|DVDRIP|BDRIP|BRRIP|BLURAY|REMUX|'
    r'X26[45]|H26[45]|HEVC|AVC|10[ ._-]?BIT|2160P|1080P|720P|480P|'
    r'PRE[ ._-]?DVD|HQ|REPACK|PROPER|INTERNAL|AMZN|NF|ESUB|DDP?\d(?:\.\d)?|'
    r'AAC|AC3|DTS)\b',
    re.IGNORECASE
)
_LANGUAGE_TOKEN_RE = re.compile(
    r'(?<![a-z])(?:' + '|'.join(sorted(map(re.escape, LANGUAGE_ALIASES), key=len, reverse=True)) + r')(?![a-z])',
    re.IGNORECASE
)

def clean_extracted_title(title: Any, year: Optional[int] = None) -> str:
    """Remove release years, episode identifiers, and common torrent tags from a title."""
    text = unicodedata.normalize('NFKC', str(title or '')).strip()
    text = re.sub(r'^www\.[^\s-]+\s*-\s*', '', text, flags=re.IGNORECASE)

    # The series title ends before its season/episode identifier. This also
    # strips an episode title accidentally included in an LLM's series title.
    episode_marker = re.search(
        r'\b(?:S\d{1,2}[\s._-]*E\d{1,3}|Season\s*\d{1,2}[\s._-]+(?:Episode|EP)\s*\d{1,3})\b',
        text,
        re.IGNORECASE
    )
    season_marker = re.search(r'\b(?:S\d{1,2}|Season\s*\d{1,2})\b', text, re.IGNORECASE)
    release_marker = _RELEASE_MARKER_RE.search(text)
    cut_points = [m.start() for m in (episode_marker, season_marker, release_marker) if m]
    if cut_points:
        text = text[:min(cut_points)]

    if year:
        text = re.sub(rf'(?<!\d){int(year)}(?!\d)', ' ', text)
    else:
        text = re.sub(r'(?<!\d)(?:19|20)\d{2}(?!\d)', ' ', text)
    text = re.sub(r'[([]\s*[)\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip(' ._,-()[]')

    # These tags are sometimes returned by the model before a quality marker.
    text = re.sub(r'\bTRUE\b\s*$', '', text, flags=re.IGNORECASE).strip(' ._,-()[]')
    # Language tags after the title are release metadata; peel them from the end.
    while text:
        match = _LANGUAGE_TOKEN_RE.search(text)
        if not match or text[match.end():].strip(' ._,-()[]'):
            break
        text = text[:match.start()].strip(' ._,-()[]')

    text = re.sub(r'\s+', ' ', text).strip(' ._,-()[]')
    return text

def clean_episode_title(title: Any) -> Optional[str]:
    """Remove release/version tags accidentally captured as an episode title."""
    text = unicodedata.normalize('NFKC', str(title or '')).strip()
    markers = [m.start() for m in (_RELEASE_MARKER_RE.search(text), re.search(r'\bv\d+\b|\brepack\b|\bproper\b', text, re.I)) if m]
    if markers:
        text = text[:min(markers)]
    text = re.sub(r'\bTRUE\b\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[._-]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip(' ._,-()[]')
    if not text or re.fullmatch(r'(?:v\d+|repack|proper|internal|complete|episode|download)', text, re.I):
        return None
    return text

def extract_filename_language(*texts: str) -> Optional[str]:
    """Use a single explicit language tag from release metadata as a fallback."""
    release_context = re.compile(
        r'(?<!\d)(?:19|20)\d{2}(?!\d)|\b(?:S\d{1,2}[\s._-]*E\d{1,3}|'
        r'(?:480|576|720|1080|1440|2160)\s*p|4k|8k|Season\s*\d+|WEB[ ._-]?(?:DL|RIP|MUX)|'
        r'HDRIP|DVDRIP|BDRIP|BLURAY|HDTV)\b',
        re.IGNORECASE
    )
    for raw_text in texts:
        text = unicodedata.normalize('NFKC', str(raw_text or ''))
        text = re.sub(r'\.(?:mkv|mp4|avi|mov|wmv|flv|m4v|webm|ts|srt|sub|vtt|ass|ssa|idx)$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^www\.[^\s-]+\s*-\s*', '', text, flags=re.IGNORECASE)
        markers = list(release_context.finditer(text))
        if not markers:
            continue
        metadata_tail = text[markers[0].end():]
        tokens = re.findall(r'[a-z]+', metadata_tail.lower())
        candidates = set()
        for token in tokens:
            language = LANGUAGE_ALIASES.get(token)
            if language:
                candidates.add(language)
        if len(candidates) == 1:
            return next(iter(candidates))
    return None

def safe_path_component(value: Any, fallback: str = 'Unknown Title') -> str:
    """Return a single portable path component for metadata-derived names."""
    text = unicodedata.normalize('NFKC', str(value or ''))
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip(' .\t\r\n')
    # Keep components below common filesystem limits while retaining the title.
    if len(text) > 180:
        text = text[:180].rstrip(' .')
    return text or fallback

def valid_year(value: Any) -> Optional[int]:
    """Coerce plausible release years and reject malformed or implausible values."""
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1888 <= year <= datetime.now().year + 3 else None

def media_folder_name(title: str, year: Optional[int]) -> str:
    title = safe_path_component(title)
    return f"{title} ({year})" if year else title

# ==============================================================================
# Helper Functions for File Inspection
# ==============================================================================
def is_sample_file(path: str, size: Optional[int] = None) -> bool:
    """Detect sample-labelled video files, regardless of their file size."""
    name = os.path.basename(path).lower()
    if os.path.splitext(name)[1] not in VIDEO_EXTENSIONS:
        return False
    stem = os.path.splitext(name)[0]
    tokens = re.findall(r'[a-z0-9]+', stem)
    quality_tags = {'480p', '720p', '1080p', '1440p', '2160p', '4k', '8k', 'x264', 'x265', 'hevc', 'web', 'webrip', 'webdl', 'bluray'}
    while tokens and tokens[-1] in quality_tags:
        tokens.pop()
    if not tokens:
        return False
    if re.fullmatch(r'sample\d*', tokens[-1]):
        return True
    return (
        len(tokens) >= 2
        and tokens[-2] == 'sample'
        and (tokens[-1] in {'clip', 'video'} or (tokens[-1].isdigit() and len(tokens[-1]) <= 3))
    )

def parse_episode_identifier(filename: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Parses season number, episode number, and optional episode title from a filename.
    Returns: (season_num, episode_num, episode_title)
    """
    name_no_ext = os.path.splitext(filename)[0]
    
    # Strip obvious marketing / URL phrases
    clean_name = re.sub(r'Online\s+for\s+Free.*', '', name_no_ext, flags=re.IGNORECASE)
    clean_name = re.sub(r'www\.[^\s]+', '', clean_name, flags=re.IGNORECASE)

    # 1. Pattern: S01E02 or S1E02 with optional title after
    m = re.search(r'\bS(\d{1,2})[\s._-]*E(\d{1,3})(?:[\s._-]+(.*?))?$', clean_name, re.IGNORECASE)
    if m:
        s_num = int(m.group(1))
        e_num = int(m.group(2))
        raw_title = m.group(3) or ''
        clean_ep_title = re.split(r'\b(1080p|720p|4k|2160p|x264|x265|hevc|web-?dl|bluray|amzn|dvdrip|hdtv|nf|esub|ddp5\.1)\b', raw_title, flags=re.IGNORECASE)[0]
        clean_ep_title = re.sub(r'[._-]+', ' ', clean_ep_title).strip()
        if re.fullmatch(r'(?:[a-z]{2,3}|english|hindi|tamil|telugu|kannada|malayalam|forced|sdh|cc|complete|episode|download|1080p|720p|2160p|4k|x26[45]|hevc|esub)', clean_ep_title, re.IGNORECASE):
            clean_ep_title = None
        return s_num, e_num, clean_ep_title if clean_ep_title else None

    # 2. Pattern: Season 2 Episode 1
    m = re.search(r'\bSeason\s*(\d{1,2})[\s._-]+(?:Episode|EP)\s*(\d{1,3})(?:[\s._-]+(.*?))?$', clean_name, re.IGNORECASE)
    if m:
        s_num = int(m.group(1))
        e_num = int(m.group(2))
        raw_title = m.group(3) or ''
        clean_ep_title = re.split(r'\b(1080p|720p|4k|2160p|x264|x265|hevc|web-?dl|bluray|amzn|dvdrip|hdtv|nf|esub|online)\b', raw_title, flags=re.IGNORECASE)[0]
        clean_ep_title = re.sub(r'[._-]+', ' ', clean_ep_title).strip()
        if re.fullmatch(r'(?:[a-z]{2,3}|english|hindi|tamil|telugu|kannada|malayalam|forced|sdh|cc|complete|episode|download|1080p|720p|2160p|4k|x26[45]|hevc|esub)', clean_ep_title, re.IGNORECASE):
            clean_ep_title = None
        return s_num, e_num, clean_ep_title if clean_ep_title else None

    # 3. Pattern: E01 or EP01, including compact names such as BankE01.
    # An embedded E marker must have at least two digits: otherwise a release
    # tag such as ZEE5 is misread as episode E5.
    m = re.search(
        r'(?:[\s._-]|\b)(?:E|EP)(\d{1,3})(?:[\s._-]+(.*?))?$|'
        r'(?<=[a-zA-Z])(?:E|EP)(\d{2,3})(?:[\s._-]+(.*?))?$',
        clean_name,
        re.IGNORECASE
    )
    if m:
        e_num = int(m.group(1) or m.group(3))
        return 1, e_num, None

    return None, None, None

def clean_media_title_heuristic(raw_name: str) -> Dict[str, Any]:
    """
    Fallback regex-based heuristic extractor if LLM is unavailable or fails.
    """
    s = re.sub(r'\.(mkv|mp4|avi|mov|wmv|flv|m4v|webm|ts|srt|sub|vtt|ass|ssa|idx)$', '', str(raw_name), flags=re.IGNORECASE).strip()
    # Strip site prefixes
    s = re.sub(r'^www\.[^\s-]+(?:\s*-\s*)?', '', s, flags=re.IGNORECASE)
    
    year = extract_release_year(s)
    parsed_season, episode, episode_title = parse_episode_identifier(s)
    season_match = re.search(r'\b(?:S|Season\s*)(\d{1,2})\b', s, flags=re.IGNORECASE)
    season = parsed_season or (int(season_match.group(1)) if season_match else None)

    is_series = bool(season or episode or re.search(r'\b(complete|s\d+|season)\b', s, flags=re.IGNORECASE))
    content_type = 'web series' if is_series else 'movie'

    # Stop at the first structural or release marker. For episode files, the
    # series title ends before SxxExx rather than including the episode name.
    cut_points = []
    episode_marker = re.search(
        r'\b(?:S\d{1,2}[\s._-]*E\d{1,3}|Season\s*\d{1,2}[\s._-]+(?:Episode|EP)\s*\d{1,3})\b',
        s,
        re.IGNORECASE
    )
    season_marker = re.search(r'\b(?:S\d{1,2}|Season\s*\d{1,2})\b', s, re.IGNORECASE)
    if episode_marker:
        cut_points.append(episode_marker.start())
    elif season_marker:
        cut_points.append(season_marker.start())
    if year:
        year_matches = list(re.finditer(rf'(?<!\d){year}(?!\d)', s))
        if year_matches:
            cut_points.append(year_matches[-1].start())
    release_marker = _RELEASE_MARKER_RE.search(s)
    if release_marker:
        cut_points.append(release_marker.start())
    base = s[:min(cut_points)] if cut_points else s
    base = re.sub(r'[._-]+', ' ', base).strip(' ._,-()[]')
    if not base:
        # Preserve numeric titles such as "1917" when no separate name exists.
        base = re.sub(r'[._-]+', ' ', s).strip(' ._,-()[]')

    if content_type == 'movie':
        cleaned = media_folder_name(base, year)
    else:
        s_str = f"S{season:02d}" if season else "S01"
        cleaned = f"{base} {s_str}E{int(episode):02d}" if episode else f"{base} {s_str}"

    return {
        'content_type': content_type,
        'title': base,
        'year': year,
        'season': season,
        'episode': episode,
        'episode_title': episode_title,
        'cleaned_name': cleaned
    }

# ==============================================================================
# LLM 1-Prompt Extractor
# ==============================================================================
SYSTEM_PROMPT = """You are an expert media library organizer.
Given a media filename or folder name from a torrent/download, analyze it and output a JSON object with:
- "content_type": "movie" | "web series" (Use episodic indicators like S01, Season 2, E04, Episode 3 to classify as web series; otherwise movie).
- "title": Clean canonical movie/series title only (properly capitalized; remove years, episode IDs, scraper sites, torrent tags, resolutions, and codecs).
- Put the year in "year". Never put SxxExx markers or an episode title in the series "title"; return episode names separately in "episode_title".
- "year": 4-digit release year as integer (e.g. 2024), or null if not present.
- "season": Season number as integer (e.g. 1 for S01), or null if movie or unspecified.
- "episode": Episode number as integer (e.g. 5 for E05), or null if whole season folder or movie.
- "episode_title": Episode title string if present (e.g. "Thursday"), or null.
- "cleaned_name": 
    - For movie: "<Title> (<Year>)" (e.g. "Inception (2010)")
    - For web series season/folder: "<Title> S<Season2Digits>" (e.g. "Kurukshetra S01")
    - For web series single episode: "<Title> S<Season2Digits>E<Episode2Digits>[ <EpisodeTitle>]" (e.g. "Special Ops S02E07 Thursday")

Use parent folder context only to resolve generic or incomplete filenames. Language tags are handled separately: metadata is preferred, and only one unambiguous source language tag is used as a fallback.

Return ONLY the JSON object. Do not include extra text or explanations."""

def extract_media_info_llm(name: str, client: Optional[Any], model_name: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Calls the local LLM in 1 single prompt to classify content type, clean title, year, season, and episode.
    Falls back gracefully to regex heuristic extractor if LLM is unavailable or fails.
    """
    if client is None:
        return clean_media_title_heuristic(name)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Media item: {name}\nParent folder context: {context or '(none)'}"}
            ],
            temperature=0.0,
            timeout=15.0
        )
        raw_output = response.choices[0].message.content or ""
        # Strip reasoning models' <think> tags if present
        if "</think>" in raw_output:
            raw_output = raw_output.split("</think>")[-1]

        # Extract JSON substring
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            # Validate required fields
            content_type = str(data.get('content_type', '')).lower()
            if 'series' in content_type:
                data['content_type'] = 'web series'
            else:
                data['content_type'] = 'movie'

            data['title'] = str(data.get('title') or '').strip()
            if not data['title'] or data['title'].lower() in {'none', 'unknown', 'untitled'}:
                return clean_media_title_heuristic(name)
            data['year'] = valid_year(data.get('year'))
            try:
                data['season'] = int(data['season']) if data.get('season') is not None else None
                data['episode'] = int(data['episode']) if data.get('episode') is not None else None
            except (TypeError, ValueError):
                return clean_media_title_heuristic(name)
            data['episode_title'] = str(data['episode_title']).strip() if data.get('episode_title') else None

            if not data.get('cleaned_name'):
                if data['content_type'] == 'movie':
                    data['cleaned_name'] = media_folder_name(data['title'], data['year'])
                else:
                    s_str = f"S{data['season']:02d}" if data['season'] else "S01"
                    data['cleaned_name'] = f"{safe_path_component(data['title'])} {s_str}"
            return data
    except Exception as e:
        logger.debug(f"LLM call fallback for '{name}': {e}")

    # Fallback heuristic
    return clean_media_title_heuristic(name)

# ==============================================================================
# Wikipedia & IMDb Direct Metadata Extraction (No DuckDuckGo)
# ==============================================================================
HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

def search_wikipedia_direct(
    base_title: str,
    year: Optional[int] = None,
    season: Optional[int] = None,
    content_type: str = 'movie'
) -> List[str]:
    """Directly queries Wikipedia's MediaWiki search API."""
    queries = []
    if season:
        queries.append(f'{base_title}')
        queries.append(f'{base_title} season {season}')
    elif content_type == 'web series':
        if year:
            queries.append(f'{base_title} {year} television series')
        queries.extend([f'{base_title} television series', f'{base_title} TV series', base_title])
    elif year:
        queries.append(f'{base_title} {year} film')
        queries.append(f'{base_title} {year}')
        queries.append(f'{base_title}')
    else:
        queries.append(f'{base_title}')

    url = 'https://en.wikipedia.org/w/api.php'
    candidates = []
    base_lower = base_title.lower()

    for q in queries:
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': q,
            'format': 'json',
            'srlimit': 5
        }
        try:
            resp = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=5)
            resp.raise_for_status()
            data = resp.json().get('query', {}).get('search', [])
            for item in data:
                title = item['title']
                t_lower = title.lower()
                # Ensure title contains main words
                if base_lower in t_lower or all(w in t_lower for w in base_lower.split() if len(w) > 2):
                    if title not in candidates:
                        candidates.append(title)
        except Exception:
            pass
        if len(candidates) >= 3:
            break
    return candidates

def extract_wikipedia_metadata(wiki_title: str, target_year: Optional[int] = None) -> Dict[str, Any]:
    """Parses Wikipedia infobox to extract original primary language and production year."""
    wiki_url = f'https://en.wikipedia.org/wiki/{urllib.parse.quote(wiki_title.replace(" ", "_"))}'
    try:
        resp = requests.get(wiki_url, headers=HTTP_HEADERS, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception:
        return {}

    infobox = soup.find('table', class_=lambda c: c and 'infobox' in c)
    if not infobox:
        return {}

    data = {}
    for row in infobox.find_all('tr'):
        th = row.find('th')
        td = row.find('td')
        if not th or not td:
            continue
        key = th.text.strip().lower()
        val = td.get_text(' ', strip=True)
        if any(k in key for k in ['directed', 'director', 'creator', 'created by']):
            data.setdefault('director', val)
        elif 'starring' in key:
            data['cast'] = [x.strip() for x in val.split(',') if x.strip()]
        elif 'language' in key:
            data['language'] = val
            data['languages'] = [x.strip() for x in val.split(',') if x.strip()]
        elif any(k in key for k in ['release date', 'original release', 'aired']):
            data['release_date'] = val
            ym = re.search(r'\b(19\d\d|20\d\d)\b', val)
            if ym:
                data['year'] = int(ym.group(1))
        elif 'country' in key:
            data['country'] = val

    # Verify year tolerance (max 1 year difference if both exist)
    if target_year and 'year' in data:
        if abs(data['year'] - target_year) > 1:
            return {}

    title_tag = soup.find('h1', id='firstHeading')
    if title_tag:
        data['title'] = title_tag.text.strip()
    data['wiki_url'] = wiki_url
    return data

def search_imdb_direct(base_title: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Directly queries IMDb's suggestion API."""
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', base_title).strip('_').lower()
    if not slug:
        return None
    url = f'https://v3.sg.media-imdb.com/suggestion/{slug[0]}/{slug}.json'
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=5)
        if resp.status_code == 200:
            items = [d for d in resp.json().get('d', []) if d.get('id', '').startswith('tt')]
            if not items:
                return None
            normalized_title = re.sub(r'[^a-z0-9]+', '', base_title.lower())
            scored = []
            for item in items:
                candidate = re.sub(r'[^a-z0-9]+', '', str(item.get('l') or '').lower())
                score = SequenceMatcher(None, normalized_title, candidate).ratio()
                if year and item.get('y'):
                    score += 0.2 if abs(int(item['y']) - year) <= 1 else -0.2
                scored.append((score, item))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            if not scored or scored[0][0] < 0.55:
                return None
            best = scored[0][1]
            return {
                'title': best.get('l'),
                'year': int(best.get('y')) if best.get('y') else None,
                'cast': [c.strip() for c in best.get('s', '').split(',') if c.strip()],
                'type': best.get('qid'),
                'imdb_id': best.get('id'),
                'imdb_url': f'https://www.imdb.com/title/{best.get("id")}/',
            }
    except Exception:
        pass
    return None

@lru_cache(maxsize=4096)
def get_media_metadata(title: str, year: Optional[int], content_type: str, client: Optional[Any] = None, model_name: str = "google/gemma-4-12b") -> Dict[str, Any]:
    """
    Fetches metadata and production language from Wikipedia, with IMDb suggestion
    data as a fallback. IMDb suggestions do not provide language, so this
    function returns Unknown unless Wikipedia explicitly reports it. The planner
    may then use one unambiguous language tag from source naming as a fallback.
    Results are cached to minimize network requests.
    """
    # Prefer a source that explicitly reports the production language.
    wiki_candidates = search_wikipedia_direct(title, year=year, content_type=content_type)
    for cand in wiki_candidates:
        meta = extract_wikipedia_metadata(cand, target_year=year)
        if meta and meta.get('language'):
            meta['primary_language'] = normalize_language_name(meta.get('language'))
            meta['source'] = 'Wikipedia'
            return meta

    # IMDb suggestion results supply title/year/cast, but not language. Keep the
    # language unknown rather than asking the LLM for a second inference call.
    imdb_meta = search_imdb_direct(title, year=year)
    if imdb_meta:
        imdb_meta['primary_language'] = 'Unknown'
        imdb_meta['source'] = 'IMDb'
        return imdb_meta

    return {
        'title': title,
        'year': year,
        'primary_language': 'Unknown',
        'source': 'No metadata match'
    }

# ==============================================================================
# Media Organization Planner & File Processor
# ==============================================================================
class MediaOrganizer:
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        dry_run: bool = False,
        generate_csv_only: bool = False,
        action: str = 'copy',
        base_url: str = 'http://192.168.0.119:1234/v1',
        model_name: str = 'google/gemma-4-12b',
        api_key: str = 'NA',
        initialize_llm: bool = True,
        parallelism: int = 1
    ):
        if parallelism < 1:
            raise ValueError("parallelism must be at least 1")

        self.source_dir = os.path.abspath(source_dir)
        self.target_dir = os.path.abspath(target_dir)
        self.dry_run = dry_run
        self.generate_csv_only = generate_csv_only
        self.action = action.lower()
        self.model_name = model_name
        self.parallelism = parallelism
        self._cleanup_safe = False

        # Initialize OpenAI client if available
        self.client = None
        if initialize_llm and openai is not None:
            try:
                test_client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=3.0)
                test_client.models.list()
                self.client = test_client
                logger.info(f"[bold cyan]Connected to LLM endpoint:[/bold cyan] {base_url} [dim]({model_name})[/dim]")
            except Exception as e:
                logger.warning(f"[bold yellow]LLM endpoint offline/unreachable ({base_url}). Using smart heuristic extraction with direct Wikipedia & IMDb metadata.[/bold yellow]")
                self.client = None

    def plan_organization(self) -> List[Dict[str, Any]]:
        """Scan recursively and plan each title independently."""
        logger.info(f"[bold green]Scanning source directory:[/bold green] {self.source_dir}")
        if not os.path.isdir(self.source_dir):
            logger.error(f"[bold red]Source directory does not exist or is not a directory:[/bold red] {self.source_dir}")
            return []

        source_real = os.path.realpath(self.source_dir)
        target_real = os.path.realpath(self.target_dir)
        try:
            overlaps = os.path.commonpath([source_real, target_real]) in {source_real, target_real}
        except ValueError:
            overlaps = False
        if overlaps:
            logger.error("[bold red]Source and target directories overlap; refusing to plan a recursive copy.[/bold red]")
            return []
        self._cleanup_safe = True

        video_files, subtitle_files = [], []
        for root, dirs, files in os.walk(self.source_dir):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            for filename in sorted(files):
                if filename.startswith('.'):
                    continue
                path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                if ext in VIDEO_EXTENSIONS and not is_sample_file(path):
                    video_files.append(path)
                elif ext in SUBTITLE_EXTENSIONS:
                    subtitle_files.append(path)

        logger.info(f"Found [bold]{len(video_files)}[/bold] video files and [bold]{len(subtitle_files)}[/bold] subtitles.")
        plan = []
        planned_files: List[Optional[List[Dict[str, Any]]]] = [None] * len(video_files)
        with ThreadPoolExecutor(max_workers=self.parallelism, thread_name_prefix="media-plan") as executor:
            futures = {
                executor.submit(self._plan_media_file, file_path, subtitle_files): index
                for index, file_path in enumerate(video_files)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing media items"):
                planned_files[futures[future]] = future.result()

        for file_operations in planned_files:
            if file_operations:
                plan.extend(file_operations)
        return self._finalize_plan(plan)

    @staticmethod
    def _generic_title(title: str) -> bool:
        normalized = re.sub(r'[^a-z0-9]+', ' ', (title or '').lower()).strip()
        return not normalized or bool(re.fullmatch(r'\d{1,4}', normalized)) or bool(re.fullmatch(
            r'(?:episode|ep)\s*\d*|(?:season|series|video|movie|part|disc|disk|cd|complete|download|untitled|unknown|extras|featurettes|samples|subtitles|subs)(?:\s+\d+)?',
            normalized
        ))

    @staticmethod
    def _stem_key(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        stem = re.sub(r'\.(?:eng?|english|hin|hindi|tam|tamil|tel|telugu|kan|kannada|mal|malayalam|spa|spanish|fre|french|ger|german|ita|italian|jpn|japanese|kor|korean|forced|sdh|cc)$', '', stem, flags=re.I)
        return re.sub(r'[^a-z0-9]+', '', stem)

    @staticmethod
    def _subtitle_language_suffix(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'\.([a-z]{2,3}|english|hindi|tamil|telugu|kannada|malayalam|spanish|french|german|italian|japanese|korean|forced|sdh|cc)$', stem, re.I)
        return f".{match.group(1).lower()}" if match else ''

    def _matching_subtitles(
        self,
        video_path: str,
        season: Optional[int],
        episode: Optional[int],
        subtitle_files: List[str]
    ) -> List[str]:
        video_key = self._stem_key(video_path)
        video_parent = os.path.dirname(video_path)
        matching = []
        for sub in subtitle_files:
            if self._stem_key(sub) == video_key:
                matching.append(sub)
                continue
            sub_season, sub_episode, _ = parse_episode_identifier(os.path.basename(sub))
            if episode is None or sub_episode != episode:
                continue
            if sub_season is not None and season is not None and sub_season != season:
                continue
            if os.path.dirname(sub) == video_parent:
                matching.append(sub)

        # A subtitle with no season marker is ambiguous if the same episode
        # number exists in multiple seasons in this source directory.
        season_marked = [s for s in matching if parse_episode_identifier(os.path.basename(s))[0] is not None]
        unmarked = [s for s in matching if parse_episode_identifier(os.path.basename(s))[0] is None]
        if season is not None and season_marked:
            matching = season_marked
        elif season is not None and len(unmarked) > 1:
            matching = [s for s in unmarked if os.path.dirname(s) == video_parent]
            if len(matching) != 1:
                matching = []
        return list(dict.fromkeys(matching))

    def _plan_media_file(self, file_path: str, subtitle_files: List[str]) -> List[Dict[str, Any]]:
        """Extract and plan one video, using folders only as context clues."""
        filename = os.path.basename(file_path)
        parent = os.path.dirname(file_path)
        relative_parent = os.path.relpath(parent, self.source_dir)
        context_parts = [] if relative_parent == '.' else relative_parent.split(os.sep)
        context = ' / '.join(context_parts)
        info = extract_media_info_llm(filename, self.client, self.model_name, context=context)
        title = str(info.get('title') or '').strip()
        year = valid_year(info.get('year')) or extract_release_year(filename)
        season, episode, episode_title = parse_episode_identifier(filename)
        season = season or info.get('season')
        episode = episode or info.get('episode')
        episode_title = clean_episode_title(episode_title or info.get('episode_title'))

        for folder in reversed(context_parts):
            season_match = re.search(r'\b(?:S|Season\s*)(\d{1,2})\b', folder, re.I)
            if season is None and season_match:
                season = int(season_match.group(1))

        # Context helps when releases use generic names such as Episode 01.mkv
        # or 01.mkv inside a season folder. It is never used to merge files.
        if self._generic_title(title):
            for folder in reversed(context_parts):
                candidate = clean_media_title_heuristic(folder)
                candidate_title = str(candidate.get('title') or '').strip()
                if not self._generic_title(candidate_title):
                    title = candidate_title
                    year = year or valid_year(candidate.get('year'))
                    season = season or candidate.get('season')
                    if candidate.get('content_type') == 'web series' or season is not None:
                        info['content_type'] = 'web series'
                    break

        stem = os.path.splitext(filename)[0]
        numeric_episode = re.fullmatch(r'(?:(?:episode|ep)[ ._-]*)?(\d{1,3})', stem, re.I)
        if episode is None and numeric_episode and season is not None:
            episode = int(numeric_episode.group(1))

        is_episode = episode is not None or bool(parse_episode_identifier(filename)[1] is not None)
        content_type = 'web series' if is_episode or info.get('content_type') == 'web series' else 'movie'
        title = clean_extracted_title(title, year=year)
        title = '' if self._generic_title(title) else title
        title_needs_review = not title
        title = safe_path_component(title)
        if title_needs_review:
            logger.warning(f"[yellow]Could not determine a title; marking for review: {file_path}[/yellow]")

        meta = get_media_metadata(title, year, content_type, self.client, self.model_name)
        primary_language = normalize_language_name(meta.get('primary_language'))
        if primary_language == 'Unknown':
            primary_language = extract_filename_language(filename, context) or 'Unknown'
        lang = safe_path_component(primary_language, 'Unknown')
        first_year = valid_year(meta.get('year')) or year
        ext = os.path.splitext(filename)[1].lower()
        ops = []

        if content_type == 'movie':
            movie_folder = media_folder_name(title, first_year)
            dest_dir = os.path.join(self.target_dir, 'Movies', lang, movie_folder)
            dest_path = os.path.join(dest_dir, f"{movie_folder}{ext}")
            op = self._make_op_dict(file_path, dest_path, dest_dir, 'movie', title, first_year, None, None, lang)
            if title_needs_review:
                op['Status'] = 'NEEDS_REVIEW_NO_TITLE'
            ops.append(op)
            matching_subtitles = self._matching_subtitles(file_path, None, None, subtitle_files)
            for sub in matching_subtitles:
                sub_ext = os.path.splitext(sub)[1].lower()
                suffix = self._subtitle_language_suffix(sub)
                sub_name = f"{movie_folder}{suffix}{sub_ext}"
                sub_op = self._make_op_dict(sub, os.path.join(dest_dir, sub_name), dest_dir, 'movie_subtitle', title, first_year, None, None, lang)
                if title_needs_review:
                    sub_op['Status'] = 'NEEDS_REVIEW_NO_TITLE'
                ops.append(sub_op)
            return ops

        season = int(season or 1)
        series_folder = media_folder_name(title, first_year)
        season_folder = f"{title} S{season:02d}"
        dest_dir = os.path.join(self.target_dir, 'Web Series', lang, series_folder, season_folder)
        if episode is None:
            # Do not silently invent E01 for a file that has no episode identity.
            dest_path = os.path.join(dest_dir, safe_path_component(stem) + ext)
            op = self._make_op_dict(file_path, dest_path, dest_dir, 'web series', title, first_year, season, None, lang)
            op['Status'] = 'NEEDS_REVIEW_NO_TITLE' if title_needs_review else 'NEEDS_REVIEW_NO_EPISODE_NUMBER'
            ops.append(op)
            return ops

        episode = int(episode)
        ep_code = f"S{season:02d}E{episode:02d}"
        clean_ep_title = safe_path_component(episode_title, '') if episode_title else ''
        ep_suffix = f" {clean_ep_title}" if clean_ep_title else ''
        dest_path = os.path.join(dest_dir, f"{title} {ep_code}{ep_suffix}{ext}")
        video_op = self._make_op_dict(file_path, dest_path, dest_dir, 'web series', title, first_year, season, episode, lang, clean_ep_title or None)
        if title_needs_review:
            video_op['Status'] = 'NEEDS_REVIEW_NO_TITLE'
        ops.append(video_op)
        for sub in self._matching_subtitles(file_path, season, episode, subtitle_files):
            sub_ext = os.path.splitext(sub)[1].lower()
            language_suffix = self._subtitle_language_suffix(sub)
            sub_path = os.path.join(dest_dir, f"{title} {ep_code}{ep_suffix}{language_suffix}{sub_ext}")
            sub_op = self._make_op_dict(sub, sub_path, dest_dir, 'series_subtitle', title, first_year, season, episode, lang, clean_ep_title or None)
            if title_needs_review:
                sub_op['Status'] = 'NEEDS_REVIEW_NO_TITLE'
            ops.append(sub_op)
        return ops

    def _plan_single_file(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Compatibility wrapper retained for callers of the earlier planner."""
        parent = os.path.dirname(file_path)
        subtitles = [os.path.join(parent, f) for f in os.listdir(parent)
                     if os.path.splitext(f)[1].lower() in SUBTITLE_EXTENSIONS]
        return self._plan_media_file(file_path, subtitles)

    def _plan_directory(self, dir_path: str, dir_name: str) -> List[Dict[str, Any]]:
        """Compatibility wrapper; recursive planning is now performed per file."""
        videos, subtitles = [], []
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for filename in files:
                path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                if ext in VIDEO_EXTENSIONS and not is_sample_file(path):
                    videos.append(path)
                elif ext in SUBTITLE_EXTENSIONS:
                    subtitles.append(path)
        result = []
        for path in sorted(videos):
            result.extend(self._plan_media_file(path, subtitles))
        return result

    @staticmethod
    def _finalize_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate repeated rows and mark every destination conflict."""
        deduplicated, seen_rows = [], set()
        for item in plan:
            row_key = (os.path.realpath(item['Source Path']), os.path.normcase(os.path.abspath(item['Target File Path'])))
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            deduplicated.append(item)

        destinations: Dict[str, List[Dict[str, Any]]] = {}
        for item in deduplicated:
            src, dst = item['Source Path'], item['Target File Path']
            if item['Status'] != 'PLANNED':
                continue
            if os.path.realpath(src) == os.path.realpath(dst):
                item['Status'] = 'SAME_SOURCE_AND_TARGET'
            elif os.path.lexists(dst):
                item['Status'] = 'TARGET_ALREADY_EXISTS'
            else:
                destinations.setdefault(os.path.normcase(os.path.abspath(dst)), []).append(item)

        for colliding in destinations.values():
            sources = {os.path.realpath(item['Source Path']) for item in colliding}
            if len(sources) > 1:
                for item in colliding:
                    item['Status'] = 'PLAN_DESTINATION_COLLISION'
        return deduplicated

    def _make_op_dict(
        self,
        src: str,
        dst: str,
        dest_dir: str,
        content_type: str,
        title: str,
        year: Optional[int],
        season: Optional[int],
        episode: Optional[int],
        lang: str,
        ep_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a standardized operation descriptor."""
        status = 'PLANNED'
        if os.path.lexists(dst):
            status = 'TARGET_ALREADY_EXISTS'

        return {
            'Source Path': src,
            'Original Name': os.path.basename(src),
            'Content Type': content_type,
            'Title': title,
            'Year': year,
            'Season': season,
            'Episode': episode,
            'Episode Title': ep_title,
            'Primary Language': lang,
            'Target Directory': dest_dir,
            'Target File Path': dst,
            'Status': status
        }

    def cleanup_source(self, dry_run: bool = False) -> Tuple[int, int]:
        """Remove source files except non-sample videos and optional subtitles."""
        if not self._cleanup_safe or not os.path.isdir(self.source_dir):
            logger.warning("[yellow]Source cleanup skipped because the source scan was not validated.[/yellow]")
            return 0, 0

        removed_files = 0
        removed_dirs = 0
        for root, dirs, files in os.walk(self.source_dir, topdown=True):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            for filename in files:
                path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                keep = ext in SUBTITLE_EXTENSIONS or (ext in VIDEO_EXTENSIONS and not is_sample_file(path))
                if keep:
                    continue
                if dry_run:
                    logger.info(f"[yellow][CLEANUP DRY-RUN][/yellow] Would delete [cyan]{path}[/cyan]")
                    removed_files += 1
                    continue
                try:
                    os.unlink(path)
                    logger.info(f"[yellow][CLEANUP][/yellow] Deleted [cyan]{path}[/cyan]")
                    removed_files += 1
                except OSError as exc:
                    logger.error(f"[red][CLEANUP ERROR][/red] Could not delete '{path}': {exc}")

        if not dry_run:
            for root, dirs, _ in os.walk(self.source_dir, topdown=False):
                for directory in dirs:
                    if directory.startswith('.'):
                        continue
                    path = os.path.join(root, directory)
                    try:
                        os.rmdir(path)
                        removed_dirs += 1
                    except OSError:
                        # Non-empty folders still contain retained media/subtitles.
                        pass

        action = 'Would remove' if dry_run else 'Removed'
        logger.info(f"[bold yellow]Source cleanup: {action} {removed_files} files and {removed_dirs} empty folders.[/bold yellow]")
        return removed_files, removed_dirs

    def load_reviewed_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load user-reviewed destinations from a generated plan CSV."""
        required_columns = {'Source Path', 'Target File Path'}
        plan = []
        source_root = os.path.realpath(self.source_dir)
        target_root = os.path.realpath(self.target_dir)

        def path_is_within(path: str, root: str) -> bool:
            try:
                return os.path.commonpath([os.path.realpath(path), root]) == root
            except ValueError:
                return False

        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as csv_file:
            reader = csv.DictReader(csv_file)
            columns = set(reader.fieldnames or [])
            missing = required_columns - columns
            if missing:
                raise ValueError(f"Reviewed CSV is missing required column(s): {', '.join(sorted(missing))}")

            for row_number, row in enumerate(reader, start=2):
                src_value = (row.get('Source Path') or '').strip()
                dst_value = (row.get('Target File Path') or '').strip()
                if not src_value or not dst_value:
                    raise ValueError(f"CSV row {row_number} must have Source Path and Target File Path")

                src = os.path.abspath(src_value if os.path.isabs(src_value) else os.path.join(self.source_dir, src_value))
                dst = os.path.abspath(dst_value if os.path.isabs(dst_value) else os.path.join(self.target_dir, dst_value))
                if not path_is_within(src, source_root):
                    raise ValueError(f"CSV row {row_number} source is outside source_dir: {src}")
                if not path_is_within(dst, target_root):
                    raise ValueError(f"CSV row {row_number} destination is outside target_dir: {dst}")
                if not os.path.isfile(src):
                    raise ValueError(f"CSV row {row_number} source file does not exist: {src}")

                item = {key: value for key, value in row.items() if key is not None}
                item['Source Path'] = src
                item['Original Name'] = os.path.basename(src)
                item['Target File Path'] = dst
                item['Target Directory'] = os.path.dirname(dst)
                item['Status'] = 'PLANNED'  # Re-evaluate conflicts against the reviewed destination.
                plan.append(item)

        if not plan:
            raise ValueError("Reviewed CSV contains no operation rows")
        return self._finalize_plan(plan)

    def _execute_file_operation(
        self,
        item: Dict[str, Any],
        skip_statuses: set
    ) -> Tuple[str, str]:
        """Copy or move one planned file from a dedicated worker thread."""
        src = item['Source Path']
        dst = item['Target File Path']
        if item['Status'] in skip_statuses:
            return 'skip', item['Status']

        created_destination = False
        try:
            os.makedirs(item['Target Directory'], exist_ok=True)
            if self.action == 'move':
                try:
                    os.link(src, dst)
                    created_destination = True
                except FileExistsError:
                    raise
                except OSError:
                    with open(src, 'rb') as source_file, open(dst, 'xb') as target_file:
                        created_destination = True
                        shutil.copyfileobj(source_file, target_file)
                    shutil.copystat(src, dst)
                os.unlink(src)
                return 'success', 'MOVED'

            with open(src, 'rb') as source_file, open(dst, 'xb') as target_file:
                created_destination = True
                shutil.copyfileobj(source_file, target_file)
            shutil.copystat(src, dst)
            return 'success', 'COPIED'
        except FileExistsError:
            return 'skip', 'Destination appeared after planning'
        except Exception as exc:
            if created_destination:
                try:
                    os.unlink(dst)
                except OSError:
                    pass
            return 'error', str(exc)

    def cleanup_sources_with_existing_targets(
        self,
        plan: List[Dict[str, Any]],
        dry_run: bool = False
    ) -> Tuple[int, int]:
        """Remove planned source files only after all their target files exist."""
        source_root = os.path.abspath(self.source_dir)
        targets_by_source: Dict[str, set] = {}
        source_paths: Dict[str, str] = {}

        for item in plan:
            src = os.path.abspath(item['Source Path'])
            dst = os.path.abspath(item['Target File Path'])
            source_key = os.path.normcase(src)
            source_paths[source_key] = src
            targets_by_source.setdefault(source_key, set()).add(dst)

        removed_files = 0
        retained_files = 0
        for source_key, targets in targets_by_source.items():
            src = source_paths[source_key]
            try:
                source_is_within_root = os.path.commonpath([src, source_root]) == source_root
            except ValueError:
                source_is_within_root = False
            if not source_is_within_root:
                logger.warning(f"[yellow][CLEANUP SKIP][/yellow] Source is outside source directory: {src}")
                retained_files += 1
                continue

            # A same-path target is already the source file; never unlink it.
            if any(os.path.realpath(src) == os.path.realpath(dst) for dst in targets):
                retained_files += 1
                continue
            if not os.path.isfile(src):
                continue
            missing_targets = [dst for dst in targets if not os.path.isfile(dst)]
            if missing_targets:
                logger.warning(
                    f"[yellow][CLEANUP SKIP][/yellow] Keeping [cyan]{src}[/cyan]; "
                    f"target file is missing: [dim]{missing_targets[0]}[/dim]"
                )
                retained_files += 1
                continue

            if dry_run:
                logger.info(f"[yellow][CLEANUP DRY-RUN][/yellow] Would delete [cyan]{src}[/cyan]")
                removed_files += 1
                continue

            try:
                os.unlink(src)
                logger.info(f"[yellow][CLEANUP][/yellow] Deleted source file [cyan]{src}[/cyan]")
                removed_files += 1
            except OSError as exc:
                logger.error(f"[red][CLEANUP ERROR][/red] Could not delete '{src}': {exc}")
                retained_files += 1

        removed_dirs = 0
        if not dry_run and os.path.isdir(self.source_dir):
            for root, dirs, _ in os.walk(self.source_dir, topdown=False):
                for directory in dirs:
                    if directory.startswith('.'):
                        continue
                    path = os.path.join(root, directory)
                    try:
                        os.rmdir(path)
                        removed_dirs += 1
                    except OSError:
                        # Keep directories that still contain files or subdirectories.
                        pass

        action = 'Would remove' if dry_run else 'Removed'
        logger.info(
            f"[bold yellow]Target-verified source cleanup: {action} {removed_files} files "
            f"and {removed_dirs} empty folders; kept {retained_files} files.[/bold yellow]"
        )
        return removed_files, removed_dirs

    def execute_plan(
        self,
        plan: List[Dict[str, Any]],
        csv_path: Optional[str] = None,
        write_csv: bool = True,
        cleanup_after_execute: bool = True,
        cleanup_copied_sources: bool = False
    ):
        """Execute operations, then remove source files that are not media/subtitles."""
        if not plan:
            logger.warning("[bold yellow]No media operations planned.[/bold yellow]")
            if not self.generate_csv_only:
                self.cleanup_source(dry_run=self.dry_run)
            return

        df = pd.DataFrame(plan)

        if write_csv:
            # Default CSV export path
            if not csv_path:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                csv_path = os.path.join(os.getcwd(), f"media_organization_plan_{ts}.csv")

            df.to_csv(csv_path, index=False)
            logger.info(f"[bold green]Saved plan CSV to:[/bold green] {csv_path}")

        # Summary Table
        table = Table(title="Media Organization Summary", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        counts = df['Content Type'].value_counts().to_dict()
        for k, v in counts.items():
            table.add_row(str(k), str(v))
        table.add_row("[bold]Total Operations[/bold]", f"[bold]{len(df)}[/bold]")
        console.print(table)

        if self.generate_csv_only:
            logger.info("[bold yellow]--generate-csv-only specified. Exiting without file modifications.[/bold yellow]")
            return

        if self.dry_run:
            logger.info("[bold yellow]--dry-run mode enabled. Showing planned actions:[/bold yellow]")
            for item in plan:
                logger.info(
                    f"[yellow][DRY-RUN][/yellow] [{item['Content Type']}] [{item['Primary Language']}] "
                    f"\"{item['Original Name']}\" -> [cyan]{item['Target File Path']}[/cyan] ({item['Status']})"
                )
            logger.info("[bold green]Dry run completed successfully. No files were modified.[/bold green]")
            if cleanup_copied_sources:
                self.cleanup_sources_with_existing_targets(plan, dry_run=True)
            self.cleanup_source(dry_run=True)
            return

        # Create destinations exclusively so a file appearing after planning
        # can never be truncated or replaced. For moves, a hard link followed
        # by unlinking the source gives a metadata-only same-filesystem move
        # without allowing an existing destination to be overwritten. If the
        # filesystem cannot hard-link the source (including cross-volume
        # moves), fall back to the existing copy-and-delete behavior.
        logger.info(
            f"[bold green]Starting {self.action.upper()} operations for {len(plan)} files "
            f"with parallelism {self.parallelism}...[/bold green]"
        )
        success_count = 0
        skip_count = 0
        error_count = 0
        skip_statuses = {
            'TARGET_ALREADY_EXISTS', 'PLAN_DESTINATION_COLLISION',
            'SAME_SOURCE_AND_TARGET', 'NEEDS_REVIEW_NO_EPISODE_NUMBER',
            'NEEDS_REVIEW_NO_TITLE',
            'SKIPPED_ALREADY_EXISTS', 'TARGET_EXISTS_DIFFERENT_SIZE', 'TARGET_EXISTS'
        }

        with ThreadPoolExecutor(max_workers=self.parallelism, thread_name_prefix="media-file") as executor:
            futures = {
                executor.submit(self._execute_file_operation, item, skip_statuses): item
                for item in plan
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{self.action.capitalize()}ing files"):
                item = futures[future]
                src = item['Source Path']
                dst = item['Target File Path']
                result, detail = future.result()
                if result == 'success':
                    logger.info(f"[green][{detail}][/green] [cyan]{item['Original Name']}[/cyan] -> [bold green]{dst}[/bold green]")
                    success_count += 1
                elif result == 'skip':
                    logger.warning(f"[yellow][SKIP][/yellow] {detail}: [dim]{dst}[/dim]")
                    skip_count += 1
                else:
                    logger.error(f"[bold red][ERROR][/bold red] Failed to {self.action} '{src}' to '{dst}': {detail}")
                    error_count += 1

        logger.info(
            f"[bold green]Execution Complete:[/bold green] "
            f"Successfully processed: [bold]{success_count}[/bold], "
            f"Skipped: [yellow]{skip_count}[/yellow], "
            f"Errors: [red]{error_count}[/red]"
        )
        if cleanup_copied_sources:
            self.cleanup_sources_with_existing_targets(plan)
        if cleanup_after_execute:
            self.cleanup_source()

# ==============================================================================
# CLI Entry Point
# ==============================================================================
def positive_int(value: str) -> int:
    """Parse a positive integer for worker count CLI options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a media organization CSV for review with --generate-csv-only, then apply your "
            "edited destinations with --apply-csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_dir", type=str, nargs="?", help="Source directory containing unorganized media")
    parser.add_argument("target_dir", type=str, nargs="?", help="Target directory where organized media will be placed")
    parser.add_argument("-s", "--source", dest="source_opt", type=str, help="Source directory (flag alias)")
    parser.add_argument("-t", "--target", dest="target_opt", type=str, help="Target directory (flag alias)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate organizing and source cleanup without modifying files")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="After execution, delete planned source files only when their target files exist; remove empty source folders"
    )
    parser.add_argument("--generate-csv-only", action="store_true", default=False, help="Only generate the plan CSV; do not modify or clean the source")
    parser.add_argument("--apply-csv", type=str, default=None, metavar="CSV", help="Apply a reviewed plan CSV; edits to Target File Path are used as the final destinations")
    parser.add_argument("--action", choices=["copy", "move"], default="move", help="File action to perform (copy or move*)")
    parser.add_argument("--csv-path", type=str, default=None, help="Custom output path for the plan CSV report")
    parser.add_argument("--base-url", type=str, default="http://192.168.0.119:1234/v1", help="Base URL for local LM Studio OpenAI API")
    parser.add_argument("--model", type=str, default="google/gemma-4-12b", help="Model key for extraction")
    parser.add_argument("--api-key", type=str, default="NA", help="API key for endpoint (default NA)")
    parser.add_argument(
        "-j", "--parallelism",
        type=positive_int,
        default=1,
        metavar="N",
        help="Number of concurrent LLM planning and file copy/move workers"
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging verbosity level")

    args = parser.parse_args()

    if args.generate_csv_only and args.apply_csv:
        parser.error("--generate-csv-only and --apply-csv cannot be used together")

    # Determine source and target directories
    source = args.source_opt or args.source_dir
    target = args.target_opt or args.target_dir

    if not source or not target:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.log_level)

    organizer = MediaOrganizer(
        source_dir=source,
        target_dir=target,
        dry_run=args.dry_run,
        generate_csv_only=args.generate_csv_only,
        action=args.action,
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.api_key,
        initialize_llm=not bool(args.apply_csv),
        parallelism=args.parallelism
    )

    if args.apply_csv:
        try:
            plan = organizer.load_reviewed_csv(args.apply_csv)
        except (OSError, ValueError, csv.Error) as exc:
            logger.error(f"[bold red]Could not load reviewed CSV:[/bold red] {exc}")
            sys.exit(2)
        organizer.execute_plan(
            plan,
            write_csv=False,
            cleanup_after_execute=False,
            cleanup_copied_sources=args.cleanup
        )
    else:
        plan = organizer.plan_organization()
        organizer.execute_plan(
            plan,
            csv_path=args.csv_path,
            cleanup_copied_sources=args.cleanup
        )

if __name__ == "__main__":
    main()
