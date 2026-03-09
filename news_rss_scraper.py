#!/usr/bin/env python3
"""
Nepal News RSS Feed Scraper

Fetches articles from 55+ Nepali news RSS feeds (national + 7 provinces). Supports both English and
Nepali language sources. Outputs structured JSON for corpus building.

Usage:
    python news_rss_scraper.py                      # Fetch all feeds
    python news_rss_scraper.py --language en         # English feeds only
    python news_rss_scraper.py --language ne         # Nepali feeds only
    python news_rss_scraper.py --feed tkp            # Single feed
    python news_rss_scraper.py --output data/        # Save JSON per feed
    python news_rss_scraper.py --output corpus.jsonl --format jsonl  # JSONL for training

Requirements:
    pip install feedparser requests
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from email.utils import parsedate_to_datetime

import feedparser
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Article:
    """A news article from an RSS feed."""
    id: str
    title: str
    url: str
    source_id: str
    source_name: str
    language: str
    published_at: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ============ Nepal News RSS Feeds ============
# Curated list of working RSS feeds as of 2025

FEEDS = {
    # ── English Sources ──
    "tkp": {
        "name": "The Kathmandu Post",
        "url": "https://kathmandupost.com/rss",
        "language": "en",
        "priority": 1,
    },
    "onlinekhabar_en": {
        "name": "OnlineKhabar (English)",
        "url": "https://english.onlinekhabar.com/feed",
        "language": "en",
        "priority": 1,
    },
    "khabarhub": {
        "name": "Khabarhub",
        "url": "https://english.khabarhub.com/feed/",
        "language": "en",
        "priority": 2,
    },
    "risingnepaldaily": {
        "name": "The Rising Nepal",
        "url": "https://risingnepaldaily.com/rss",
        "language": "en",
        "priority": 2,
    },
    "spotlight": {
        "name": "Spotlight Nepal",
        "url": "https://www.spotlightnepal.com/feed/",
        "language": "en",
        "priority": 3,
    },
    "bbc_south_asia": {
        "name": "BBC South Asia",
        "url": "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
        "language": "en",
        "priority": 1,
    },

    # ── Nepali (Devanagari) Sources ──
    "onlinekhabar_ne": {
        "name": "OnlineKhabar (Nepali)",
        "url": "https://www.onlinekhabar.com/feed",
        "language": "ne",
        "priority": 1,
    },
    "setopati": {
        "name": "Setopati",
        "url": "https://www.setopati.com/feed",
        "language": "ne",
        "priority": 1,
    },
    "nagariknews": {
        "name": "Nagarik News",
        "url": "https://nagariknews.nagariknetwork.com/feed",
        "language": "ne",
        "priority": 1,
    },
    "annapurnapost": {
        "name": "Annapurna Post",
        "url": "https://annapurnapost.com/rss",
        "language": "ne",
        "priority": 1,
    },
    "bbc_nepali": {
        "name": "BBC Nepali",
        "url": "https://feeds.bbci.co.uk/nepali/rss.xml",
        "language": "ne",
        "priority": 1,
    },
    "gorkhapatra": {
        "name": "Gorkhapatra",
        "url": "https://gorkhapatraonline.com/rss",
        "language": "ne",
        "priority": 2,
    },
    "khabarhub_ne": {
        "name": "Khabarhub (Nepali)",
        "url": "https://khabarhub.com/feed/",
        "language": "ne",
        "priority": 2,
    },
    "himalpress": {
        "name": "Himal Press",
        "url": "https://www.himalpress.com/feed/",
        "language": "ne",
        "priority": 2,
    },
    "pahilopost": {
        "name": "Pahilo Post",
        "url": "https://pahilopost.com/feed",
        "language": "ne",
        "priority": 2,
    },
    "ap1tv": {
        "name": "AP1 TV",
        "url": "https://ap1hdtv.com/feed/",
        "language": "ne",
        "priority": 2,
    },
    "imagetv": {
        "name": "Image Channel",
        "url": "https://www.imagekhabar.com/feed",
        "language": "ne",
        "priority": 2,
    },

    # ── Economic/Finance ──
    "merolagani": {
        "name": "Mero Lagani",
        "url": "https://merolagani.com/rss.aspx",
        "language": "ne",
        "priority": 2,
    },

    # ── Koshi Province ──
    "koshi_onlinekhabar": {
        "name": "OnlineKhabar Koshi",
        "url": "https://www.onlinekhabar.com/content/province/koshi/feed",
        "language": "ne",
        "priority": 3,
    },
    "biratnagar_ob": {
        "name": "Our Biratnagar",
        "url": "https://ourbiratnagar.net/feed/",
        "language": "ne",
        "priority": 3,
    },

    # ── Madhesh Province ──
    "madhesh_onlinekhabar": {
        "name": "OnlineKhabar Madhesh",
        "url": "https://www.onlinekhabar.com/content/province/madhesh/feed",
        "language": "ne",
        "priority": 3,
    },
    "janakpur_today": {
        "name": "Janakpur Today",
        "url": "https://ejanakpurtoday.com/feed/",
        "language": "ne",
        "priority": 3,
    },

    # ── Bagmati Province ──
    "bagmati_onlinekhabar": {
        "name": "OnlineKhabar Bagmati",
        "url": "https://www.onlinekhabar.com/content/province/bagmati/feed",
        "language": "ne",
        "priority": 3,
    },
    "himalpress_bagmati": {
        "name": "HimalPress Bagmati",
        "url": "https://en.himalpress.com/category/province/province-3/feed/",
        "language": "en",
        "priority": 3,
    },

    # ── Gandaki Province ──
    "gandaki_onlinekhabar": {
        "name": "OnlineKhabar Gandaki",
        "url": "https://www.onlinekhabar.com/content/province/gandaki/feed",
        "language": "ne",
        "priority": 3,
    },
    "gandaknews": {
        "name": "Gandak News",
        "url": "https://www.gandaknews.com/feed/",
        "language": "ne",
        "priority": 3,
    },
    "pokharahotline": {
        "name": "Pokhara Hotline",
        "url": "https://pokharahotline.com/feed/",
        "language": "ne",
        "priority": 4,
    },

    # ── Lumbini Province ──
    "lumbini_onlinekhabar": {
        "name": "OnlineKhabar Lumbini",
        "url": "https://www.onlinekhabar.com/content/province/lumbini/feed",
        "language": "ne",
        "priority": 3,
    },
    "lumbini_online": {
        "name": "Lumbini Online",
        "url": "https://www.lumbinionline.com/feed/",
        "language": "ne",
        "priority": 3,
    },

    # ── Karnali Province ──
    "karnali_onlinekhabar": {
        "name": "OnlineKhabar Karnali",
        "url": "https://www.onlinekhabar.com/content/province/karnali/feed",
        "language": "ne",
        "priority": 3,
    },
    "karnali_mission": {
        "name": "Karnali Mission",
        "url": "https://karnalimission.com/feed/",
        "language": "ne",
        "priority": 3,
    },

    # ── Sudurpashchim Province ──
    "sudurpashchim_onlinekhabar": {
        "name": "OnlineKhabar Sudurpashchim",
        "url": "https://www.onlinekhabar.com/content/province/sudurpashchim/feed",
        "language": "ne",
        "priority": 3,
    },
}

# HTML-only sources (no RSS, need dedicated scrapers — listed here for reference)
HTML_ONLY_SOURCES = {
    "himalayan": {"name": "The Himalayan Times", "url": "https://www.thehimalayantimes.com/", "language": "en"},
    "republica": {"name": "My Republica", "url": "https://myrepublica.nagariknetwork.com/", "language": "en"},
    "ekantipur": {"name": "eKantipur", "url": "https://ekantipur.com/", "language": "ne", "note": "Use ekantipur_scraper.py"},
    "ratopati": {"name": "Ratopati", "url": "https://ratopati.com/", "language": "ne"},
    "nepalitimes": {"name": "Nepali Times", "url": "https://www.nepalitimes.com/", "language": "en"},
    "kantipurtv": {"name": "Kantipur TV", "url": "https://www.kantipurtv.com/", "language": "ne"},
    "koshi_ratopati": {"name": "Ratopati Koshi", "url": "https://koshi.ratopati.com/", "language": "ne"},
    "madhesh_ratopati": {"name": "Ratopati Madhesh", "url": "https://madhesh.ratopati.com/", "language": "ne"},
    "bagmati_ratopati": {"name": "Ratopati Bagmati", "url": "https://bagmati.ratopati.com/", "language": "ne"},
    "gandaki_ratopati": {"name": "Ratopati Gandaki", "url": "https://gandaki.ratopati.com/province/4", "language": "ne"},
    "lumbini_ratopati": {"name": "Ratopati Lumbini", "url": "https://lumbini.ratopati.com/", "language": "ne"},
    "karnali_ratopati": {"name": "Ratopati Karnali", "url": "https://karnali.ratopati.com/", "language": "ne"},
    "sudurpashchim_ratopati": {"name": "Ratopati Sudurpashchim", "url": "https://sudurpashchim.ratopati.com/", "language": "ne"},
}


def clean_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_feed(feed_id: str, feed_config: dict, timeout: int = 30) -> List[Article]:
    """Fetch and parse a single RSS feed."""
    url = feed_config["url"]
    logger.info(f"Fetching {feed_config['name']} ({url})")

    try:
        response = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; NepaliCorpus/1.0)"
        })
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch {feed_id}: {e}")
        return []

    feed = feedparser.parse(response.content)
    articles = []

    for entry in feed.entries:
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        if not title or not link:
            continue

        # Parse published date
        published_at = None
        for date_field in ["published", "updated", "created"]:
            raw = entry.get(date_field)
            if raw:
                try:
                    published_at = parsedate_to_datetime(raw).isoformat()
                except Exception:
                    pass
                break

        # Extract content/summary
        summary = clean_html(entry.get("summary", ""))
        content = None
        if hasattr(entry, "content") and entry.content:
            content = clean_html(entry.content[0].get("value", ""))

        # Categories/tags
        categories = [t.get("term", "") for t in entry.get("tags", []) if t.get("term")]

        article_id = entry.get("id", link)

        articles.append(Article(
            id=article_id,
            title=title,
            url=link,
            source_id=feed_id,
            source_name=feed_config["name"],
            language=feed_config["language"],
            published_at=published_at,
            summary=summary if summary else None,
            content=content if content else None,
            author=entry.get("author"),
            categories=categories,
        ))

    logger.info(f"  {feed_config['name']}: {len(articles)} articles")
    return articles


def main():
    parser = argparse.ArgumentParser(description="Scrape Nepal news RSS feeds")
    parser.add_argument("--feed", "-f", help="Specific feed ID (e.g. tkp, setopati)")
    parser.add_argument("--language", "-l", choices=["en", "ne"], help="Filter by language")
    parser.add_argument("--output", "-o", help="Output path (directory for JSON, or file for JSONL)")
    parser.add_argument("--format", choices=["json", "jsonl"], default="json", help="Output format (default: json)")
    parser.add_argument("--list", action="store_true", help="List all available feeds")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    args = parser.parse_args()

    if args.list:
        print("=" * 70)
        print("Nepal News RSS Feeds")
        print(f"Total: {len(FEEDS)} feeds")
        print("=" * 70)
        for fid, cfg in sorted(FEEDS.items(), key=lambda x: (x[1]["language"], x[1]["priority"])):
            lang_tag = "[EN]" if cfg["language"] == "en" else "[NE]"
            print(f"  {lang_tag} {fid:20s} {cfg['name']}")
        return

    # Select feeds
    if args.feed:
        if args.feed not in FEEDS:
            print(f"Unknown feed: {args.feed}. Use --list to see available feeds.")
            return
        targets = {args.feed: FEEDS[args.feed]}
    elif args.language:
        targets = {k: v for k, v in FEEDS.items() if v["language"] == args.language}
    else:
        targets = FEEDS

    print(f"Fetching {len(targets)} feeds...")

    all_articles = []
    for feed_id, feed_config in targets.items():
        articles = fetch_feed(feed_id, feed_config)
        all_articles.extend(articles)
        time.sleep(args.delay)

    print(f"\nTotal articles: {len(all_articles)}")
    en_count = sum(1 for a in all_articles if a.language == "en")
    ne_count = sum(1 for a in all_articles if a.language == "ne")
    print(f"  English: {en_count}, Nepali: {ne_count}")

    if not args.output:
        # Print sample to stdout
        for a in all_articles[:10]:
            lang = "[EN]" if a.language == "en" else "[NE]"
            print(f"\n{lang} {a.source_name}")
            print(f"  {a.title}")
            print(f"  {a.url}")
            if a.published_at:
                print(f"  {a.published_at}")
        if len(all_articles) > 10:
            print(f"\n... and {len(all_articles) - 10} more articles")
        print("\nUse --output <path> to save results.")
        return

    # Save output
    if args.format == "jsonl":
        with open(args.output, "w", encoding="utf-8") as f:
            for a in all_articles:
                f.write(json.dumps(asdict(a), ensure_ascii=False, default=str) + "\n")
        print(f"Saved {len(all_articles)} articles to {args.output}")
    else:
        if os.path.isdir(args.output) or args.output.endswith("/"):
            os.makedirs(args.output, exist_ok=True)
            # Group by source
            by_source = {}
            for a in all_articles:
                by_source.setdefault(a.source_id, []).append(a)
            for source_id, articles in by_source.items():
                path = os.path.join(args.output, f"{source_id}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([asdict(a) for a in articles], f, ensure_ascii=False, indent=2, default=str)
                print(f"  {source_id}: {len(articles)} articles → {path}")
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump([asdict(a) for a in all_articles], f, ensure_ascii=False, indent=2, default=str)
            print(f"Saved {len(all_articles)} articles to {args.output}")


if __name__ == "__main__":
    main()
