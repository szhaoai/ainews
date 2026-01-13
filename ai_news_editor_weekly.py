#!/usr/bin/env python3
"""
AI News Impact Editor
Generate a weekly digest of the most influential AI news.
Uses RSS feeds, web scraping, and algorithmic clustering/scoring.

Usage:
    python ai_news_editor.py [--days N] [--output FILE] [--max-stories N]
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from enum import Enum
from html import unescape
from typing import Optional

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'rss_sources': {
        'OpenAI Blog': 'https://openai.com/blog/rss.xml',
        'Google AI Blog': 'http://googleaiblog.blogspot.com/atom.xml',
        'Microsoft AI': 'https://blogs.microsoft.com/ai/feed/',
        'TechCrunch': 'https://techcrunch.com/feed/',
        'The Verge': 'https://www.theverge.com/rss/index.xml',
        'Wired': 'https://www.wired.com/feed/rss',
        'MIT Technology Review AI': 'https://www.technologyreview.com/topic/artificial-intelligence/feed',
        'Ars Technica AI': 'https://feeds.arstechnica.com/arstechnica/technology-lab',
        'arXiv AI': 'http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending',
        'arXiv ML': 'http://export.arxiv.org/api/query?search_query=cat:cs.LG&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending',
        'EU AI Act': 'https://artificialintelligenceact.eu/feed/',
    },
    'search_keywords': ['AI', 'artificial intelligence', 'machine learning', 'LLM', 'GPT', 'Claude', 'Llama', 'model release', 'AI regulation', 'AI safety'],
    'excluded_domains': ['spam', 'ads', 'promoted'],
    'similarity_threshold': 0.65,
    'max_candidates': 100,
    'min_impact_score': 20,
    'max_stories': 8,
    'fetch_timeout': 15,
    'request_delay': 1.0,
}

class Category(Enum):
    MODEL_RELEASE = "Model Release"
    RESEARCH = "Research"
    POLICY = "Policy"
    SECURITY = "Security"
    CHIPS_INFRA = "Chips/Infra"
    BUSINESS = "Business"
    OPEN_SOURCE = "Open Source"
    OTHER = "Other"

class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class RawCandidate:
    title: str
    outlet: str
    url: str
    published_at: datetime
    snippet: str
    relevance_score: float = 0.0
    source_type: str = "rss"

@dataclass
class StoryCluster:
    cluster_id: str
    headline: str
    category: Category
    articles: list = field(default_factory=list)
    impact_score: int = 0
    what_happened: str = ""
    why_it_matters: list = field(default_factory=list)
    what_to_watch: list = field(default_factory=list)
    key_numbers: list = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    confidence: Confidence = Confidence.MEDIUM

class RSSFetcher:
    def __init__(self, timeout=15, delay=1.0):
        self.timeout, self.delay = timeout, delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    async def fetch_feed(self, url, source_name):
        await asyncio.sleep(self.delay)
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            return self._parse_feed(r.text, url, source_name)
        except Exception as e:
            logger.error(f"Error fetching {source_name}: {e}")
            return []
    
    def _parse_feed(self, content, url, source_name):
        candidates, soup = [], BeautifulSoup(content, 'xml')
        for item in soup.find_all('item') or soup.find_all('entry'):
            try:
                title = item.find('title')
                if not title: continue
                link = item.find('link')
                if not link: continue
                pub_date = self._parse_date(item) or datetime.now(timezone.utc)
                summary = item.find('description') or item.find('summary')
                snippet = self._clean_text(summary.get_text()[:300]) if summary else ""
                candidates.append(RawCandidate(title=title.get_text().strip(), outlet=source_name, url=link.get_text().strip(), published_at=pub_date, snippet=snippet))
            except: pass
        return candidates
    
    def _parse_date(self, element):
        for tag in ['pubDate', 'published']:
            elem = element.find(tag)
            if elem and elem.string:
                for fmt in ['%Y-%m-%dT%H:%M:%SZ', '%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%d']:
                    try: dt = datetime.strptime(elem.string.strip(), fmt)
                    except: continue
                    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        return None
    
    def _clean_text(self, text):
        return re.sub(r'\s+', ' ', unescape(re.sub(r'<[^>]+>', '', text or ''))).strip()

class StoryClusterer:
    def __init__(self, threshold=0.65):
        self.threshold = threshold
    
    def cluster(self, candidates):
        if not candidates: return []
        sorted_cands = sorted(candidates, key=lambda x: x.published_at, reverse=True)
        clusters, used = [], set()
        for i, c in enumerate(sorted_cands):
            if i in used: continue
            similar = [i]
            for j in range(i+1, len(sorted_cands)):
                if j in used: continue
                if self._similarity(c, sorted_cands[j]) >= self.threshold:
                    similar.append(j)
            if len(similar) > 0:
                clusters.append(self._create_cluster(sorted_cands, similar))
                used.update(similar)
        return sorted(clusters, key=lambda x: len(x.articles), reverse=True)
    
    def _similarity(self, a, b):
        t = SequenceMatcher(None, a.title.lower(), b.title.lower()).ratio()
        s = SequenceMatcher(None, a.snippet.lower(), b.snippet.lower()).ratio() if a.snippet and b.snippet else 0
        return t * 0.6 + s * 0.3
    
    def _create_cluster(self, candidates, indices):
        primary = candidates[indices[0]]
        return StoryCluster(cluster_id='-'.join(re.sub(r'[^a-z0-9]', '', primary.title.lower()).split()[:6]),
                           headline=primary.title, category=self._categorize(primary.title, primary.snippet),
                           articles=[{'title': candidates[i].title, 'outlet': candidates[i].outlet, 'url': candidates[i].url,
                                     'published_at': candidates[i].published_at.isoformat()} for i in indices])
    
    def _categorize(self, title, snippet):
        text = (title + ' ' + snippet).lower()
        if re.search(r'\b(gpt|claude|llama|gemini|release|launch|model)\b', text): return Category.MODEL_RELEASE
        if re.search(r'\b(paper|arxiv|research|study|benchmark|dataset)\b', text): return Category.RESEARCH
        if re.search(r'\b(regulation|policy|governance|eu ai act|lawsuit)\b', text): return Category.POLICY
        if re.search(r'\b(security|vulnerability|attack|breach|safety)\b', text): return Category.SECURITY
        if re.search(r'\b(chip|gpu|nvidia|hardware|compute|tpu)\b', text): return Category.CHIPS_INFRA
        if re.search(r'\b(investment|funding|acquisition|partnership|buy|acquire)\b', text): return Category.BUSINESS
        if re.search(r'\b(open source|github)\b', text): return Category.OPEN_SOURCE
        if re.search(r'\b(ai|artificial intelligence|machine learning|llm)\b', text): return Category.RESEARCH
        return Category.OTHER

class ImpactScorer:
    SOURCE_WEIGHTS = {'openai.com': 1.0, 'arxiv.org': 1.0, 'google.com': 0.9, 'microsoft.com': 0.9, 'github.com': 0.9,
                      'techcrunch.com': 0.7, 'theverge.com': 0.7, 'wired.com': 0.7, 'technologyreview.com': 0.8}
    
    def score(self, cluster):
        s = min(self._real_impact(cluster), 30)
        s += min(self._breadth(cluster), 20)
        s += min(self._credibility(cluster), 15)
        s += min(self._novelty(cluster), 15)
        s += min(self._second_order(cluster), 20)
        return min(s, 100)
    
    def _real_impact(self, c):
        score = sum(5 for kw in ['breakthrough', 'first', 'major', 'significant'] if kw in c.headline.lower())
        if re.search(r'\d+%', c.headline): score += 10
        if c.category in [Category.MODEL_RELEASE, Category.SECURITY, Category.CHIPS_INFRA]: score += 10
        return min(score, 30)
    
    def _breadth(self, c):
        text = c.headline.lower()
        return min(sum(4 for terms in [['user', 'consumer'], ['enterprise', 'business'], ['developer'], ['government']] if any(t in text for t in terms)), 20)
    
    def _credibility(self, c):
        max_w = 0
        for a in c.articles:
            d = urlparse(a.get('url', '')).netloc.lower().replace('www.', '')
            for src, w in self.SOURCE_WEIGHTS.items():
                if src in d: max_w = max(max_w, w); break
        return int(max_w * 15)
    
    def _novelty(self, c):
        score = 5 if any(w in c.headline.lower() for w in ['new', 'first', 'breakthrough', 'latest', 'introduce']) else 0
        if c.category == Category.RESEARCH: score += 5
        return min(score, 15)
    
    def _second_order(self, c):
        text = c.headline.lower()
        score = 0
        for terms in [['regulation', 'policy'], ['safety', 'risk'], ['ecosystem', 'platform'], ['supply', 'semiconductor']]:
            if any(t in text for t in terms): score += 5
        return min(score, 20)

class EditorialProcessor:
    def process(self, clusters, min_score):
        for c in clusters:
            c.impact_score = ImpactScorer().score(c)
            c.what_happened = self._summarize(c)
            c.why_it_matters = self._why_matters(c)
            c.what_to_watch = self._watch_points(c)
            c.sources = {'primary': [{'title': a['title'], 'url': a['url'], 'publisher': a['outlet']} for a in c.articles[:1]],
                        'secondary': [{'title': a['title'], 'url': a['url'], 'publisher': a['outlet']} for a in c.articles[1:4]]}
            c.confidence = Confidence.HIGH if len(c.articles) >= 2 else Confidence.MEDIUM
        return [c for c in sorted(clusters, key=lambda x: x.impact_score, reverse=True) if c.impact_score >= min_score][:8]
    
    def _summarize(self, c):
        h = c.headline
        if c.category == Category.MODEL_RELEASE: return f"A new AI model was announced with enhanced capabilities."
        if c.category == Category.RESEARCH: return f"Researchers published new findings advancing AI."
        if c.category == Category.POLICY: return f"Policymakers took action affecting AI development."
        if c.category == Category.SECURITY: return f"A security vulnerability was identified in AI systems."
        return h[:150] + "..." if len(h) > 150 else h
    
    def _why_matters(self, c):
        bullets = {"Model Release": ["Introduces new capabilities", "May lower costs or improve performance"],
                   "Research": ["Advances scientific understanding", "Enables new applications"],
                   "Policy": ["Sets regulatory precedent", "Affects compliance requirements"],
                   "Security": ["Exposes risks to deployed systems", "Requires immediate attention"],
                   "Chips/Infra": ["Addresses compute constraints", "May reduce deployment costs"],
                   "Business": ["Signals market trends", "May accelerate enterprise adoption"]}
        return bullets.get(c.category.value, ["Notable development in AI"])[:3]
    
    def _watch_points(self, c):
        return ["Adoption rate and real-world deployment", "Community and developer response"] + \
               (["Regulatory response"] if c.category == Category.POLICY else []) + \
               (["Benchmark results"] if c.category == Category.MODEL_RELEASE else [])

class AINewsEditor:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.rss = RSSFetcher(self.config['fetch_timeout'], self.config['request_delay'])
    
    async def generate_report(self, start_time=None, end_time=None, max_stories=None):
        end_time = end_time or datetime.now(timezone.utc)
        start_time = start_time or end_time - timedelta(weeks=1)
        max_stories = max_stories or self.config['max_stories']
        
        logger.info(f"Generating report: {start_time} to {end_time}")
        candidates = []
        for name, url in self.config['rss_sources'].items():
            candidates.extend(await self.rss.fetch_feed(url, name))
        
        logger.info(f"Collected {len(candidates)} candidates")
        filtered = [c for c in candidates if start_time <= c.published_at <= end_time and self._is_relevant(c)]
        
        clusters = StoryClusterer().cluster(filtered)
        stories = EditorialProcessor().process(clusters, self.config.get('min_impact_score', 20))
        
        thesis = self._generate_thesis(stories)
        return {"date_range": {"start_iso": start_time.isoformat(), "end_iso": end_time.isoformat(), "timezone": "UTC"},
                "editorial_thesis": thesis, "method": {"deduping": "Text similarity clustering", "ranking": "Impact scoring (0-100)", "verification": "Source credibility weighting"},
                "top_stories": [{"rank": i+1, "story_id": s.cluster_id, "headline": s.headline, "impact_score": s.impact_score,
                                "category": s.category.value, "what_happened": s.what_happened, "why_it_matters": s.why_it_matters,
                                "what_to_watch": s.what_to_watch, "confidence": s.confidence.value, "sources": s.sources} for i, s in enumerate(stories[:max_stories])],
                "notable_mentions": [{"headline": c.headline, "one_line": c.what_happened, "category": c.category.value} 
                                    for c in clusters[len(stories):len(stories)+5] if hasattr(c, 'what_happened')]}
    
    def _is_relevant(self, c):
        text = (c.title + ' ' + c.snippet).lower()
        return any(kw.lower() in text for kw in self.config['search_keywords'])
    
    def _generate_thesis(self, stories):
        if not stories: return "No significant AI news stories identified."
        cats = [s.category for s in stories]
        dominant = max(set(cats), key=cats.count)
        return f"AI news dominated by {dominant.value.lower()} stories."

def _get_previous_reports(current_date_str):
    """Get list of previous report files sorted by date descending."""
    import glob
    import os
    pattern = "ai_news_report_*.html"
    files = glob.glob(pattern)
    reports = []
    for f in files:
        # Extract date from filename ai_news_report_YYYY_MM_DD.html
        match = re.search(r'ai_news_report_(\d{4})_(\d{2})_(\d{2})\.html', f)
        if match:
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            if date_str != current_date_str:
                reports.append((date_str, f))
    # Sort by date descending
    reports.sort(key=lambda x: x[0], reverse=True)
    return reports

def _generate_html_report(report, is_index=False, current_date_str=None):
    """Generate HTML version of the report for web publishing."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI News Impact Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
        .header .thesis {{ font-size: 16px; opacity: 0.9; font-style: italic; }}
        .meta {{ display: flex; gap: 20px; margin-top: 15px; font-size: 14px; opacity: 0.8; }}
        .story {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .story-rank {{ display: inline-block; background: #e74c3c; color: white; width: 32px; height: 32px; border-radius: 50%; text-align: center; line-height: 32px; font-weight: bold; margin-right: 12px; }}
        .story h2 {{ display: inline; font-size: 18px; margin: 0; color: #2c3e50; }}
        .story-meta {{ margin: 12px 0; font-size: 13px; color: #7f8c8d; }}
        .impact-badge {{ background: #27ae60; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; margin-left: 10px; }}
        .category {{ background: #3498db; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; text-transform: uppercase; }}
        .confidence-HIGH {{ color: #27ae60; font-weight: 600; }}
        .confidence-MEDIUM {{ color: #f39c12; font-weight: 600; }}
        .confidence-LOW {{ color: #e74c3c; font-weight: 600; }}
        .section {{ margin: 15px 0; }}
        .section h3 {{ font-size: 14px; color: #34495e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .section ul {{ margin: 0; padding-left: 20px; }}
        .section li {{ margin: 6px 0; color: #555; font-size: 14px; line-height: 1.5; }}
        .sources {{ background: #f8f9fa; padding: 12px; border-radius: 6px; font-size: 12px; }}
        .sources a {{ color: #3498db; text-decoration: none; }}
        .sources a:hover {{ text-decoration: underline; }}
        .method {{ background: #ecf0f1; padding: 15px; border-radius: 8px; font-size: 12px; color: #7f8c8d; margin-bottom: 20px; }}
        .notable {{ background: white; border-left: 4px solid #3498db; padding: 15px; margin-bottom: 10px; border-radius: 0 8px 8px 0; }}
        .notable h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #2c3e50; }}
        .notable p {{ margin: 0; font-size: 13px; color: #555; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 AI News Impact Report</h1>
        <p class="thesis">{report['editorial_thesis']}</p>
        <div class="meta">
            <span>📅 {report['date_range']['start_iso'][:10]} to {report['date_range']['end_iso'][:10]}</span>
            <span>📰 {len(report['top_stories'])} Top Stories</span>
        </div>
    </div>
    
    <div class="method">
        <strong>Methodology:</strong> {report['method']['deduping']} | {report['method']['ranking']} | {report['method']['verification']}
    </div>
"""
    
    for story in report['top_stories']:
        html += f"""
    <div class="story">
        <span class="story-rank">{story['rank']}</span>
        <h2>{story['headline']}</h2>
        <div class="story-meta">
            <span class="category">{story['category']}</span>
            <span class="impact-badge">Impact: {story['impact_score']}</span>
            <span class="confidence-{story['confidence']}">Confidence: {story['confidence']}</span>
        </div>
        <p><strong>What happened:</strong> {story['what_happened']}</p>
        <div class="section">
            <h3>Why it matters</h3>
            <ul>
"""
        for bullet in story['why_it_matters']:
            html += f"                <li>{bullet}</li>\n"
        html += """            </ul>
        </div>
        <div class="section">
            <h3>What to watch</h3>
            <ul>
"""
        for bullet in story['what_to_watch']:
            html += f"                <li>{bullet}</li>\n"
        html += """            </ul>
        </div>
        <div class="sources">
            <strong>Sources:</strong>
"""
        for src in story['sources'].get('primary', []):
            html += f'            <a href="{src["url"]}" target="_blank">{src["title"]}</a> ({src["publisher"]}) | \n'
        for src in story['sources'].get('secondary', []):
            html += f'            <a href="{src["url"]}" target="_blank">{src["title"]}</a> ({src["publisher"]}) | \n'
        html += """        </div>
    </div>
"""
    
    if report.get('notable_mentions'):
        html += """
    <h2 style="color: #2c3e50; margin-top: 40px;">Notable Mentions</h2>
"""
        for mention in report['notable_mentions']:
            html += f"""
    <div class="notable">
        <h3>{mention['headline']}</h3>
        <p>{mention['one_line']} <span class="category">{mention['category']}</span></p>
    </div>
"""

    if is_index and current_date_str:
        prev_reports = _get_previous_reports(current_date_str)
        if prev_reports:
            html += """
    <h2 style="color: #2c3e50; margin-top: 40px;">Previous Reports</h2>
    <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
        <ul style="list-style: none; padding: 0;">
"""
            for date_str, filename in prev_reports:
                html += f'            <li style="margin: 8px 0;"><a href="{filename}" style="color: #3498db; text-decoration: none; font-weight: 500;">📄 {date_str}</a></li>\n'
            html += """        </ul>
    </div>
"""

    html += """
</body>
</html>"""
    return html

async def main():
    args = argparse.ArgumentParser(description='AI News Impact Editor')
    args.add_argument('--days', type=float, default=7)
    args.add_argument('--output', default='ai_news_report.json')
    args.add_argument('--html-output', default='ai_news_report.html')
    args.add_argument('--max-stories', type=int, default=8)
    a = args.parse_args()
    
    # Generate date-stamped filename if using default
    date_str = datetime.now().strftime('%Y_%m_%d')
    if a.html_output == 'ai_news_report.html':
        a.html_output = f'ai_news_report_{date_str}.html'
    if a.output == 'ai_news_report.json':
        a.output = f'ai_news_report_{date_str}.json'
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=a.days)
    report = await AINewsEditor().generate_report(start_time=start_time, end_time=end_time, max_stories=a.max_stories)
    
    # Save JSON
    with open(a.output, 'w') as f: json.dump(report, f, indent=2)
    
    # Save HTML
    html_content = _generate_html_report(report)
    with open(a.html_output, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Generate index.html with previous reports section
    index_html_content = _generate_html_report(report, is_index=True, current_date_str=date_str.replace('_', '-'))
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(index_html_content)

    print(f"\n=== AI NEWS IMPACT REPORT ===")
    print(f"Thesis: {report['editorial_thesis']}")
    print(f"Top Stories: {len(report['top_stories'])}")
    for s in report['top_stories'][:5]:
        print(f"{s['rank']}. {s['headline'][:60]}... (Impact: {s['impact_score']})")
    print(f"\nJSON report: {a.output}")
    print(f"HTML report: {a.html_output}")
    print(f"GitHub Pages index: index.html")

if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: sys.exit(1)
    except Exception as e: logger.error(f"Fatal error: {e}"); raise
