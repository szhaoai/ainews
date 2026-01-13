# AI News Impact Report

Weekly AI news digest with impact scoring and categorization.
https://szhaoai.github.io/ainews/

## GitHub Pages Setup

This repository is configured for GitHub Pages. To enable public access:

1. Create a new repository on GitHub
2. Push this code to your repository:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin master
   ```
3. Go to your repository settings
4. Scroll to "Pages" section
5. Under "Source", select "Deploy from a branch"
6. Under "Branch", select "master" and "/ (root)"
7. Click "Save"
8. Your site will be available at: `https://yourusername.github.io/your-repo-name/`

## Running Locally

```bash
python ai_news_editor_weekly.py
```

This generates:
- `ai_news_report_YYYY_MM_DD.html` - Dated report
- `index.html` - Latest report with navigation to previous reports
- `ai_news_report_YYYY_MM_DD.json` - JSON data

## Features

- RSS feed aggregation from major AI sources
- Impact scoring (0-100) based on novelty, credibility, and breadth
- Automatic story clustering and deduplication
- Weekly HTML reports ready for web publishing
- Previous reports navigation in index.html
