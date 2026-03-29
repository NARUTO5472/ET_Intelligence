"""
ingestion/rss_fetcher.py
Fault-tolerant ET RSS ingestion pipeline.
Uses feedparser + newspaper3k for lightweight DOM extraction (no headless browser).
"""

from __future__ import annotations
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup

from config import ET_RSS_FEEDS, MAX_ARTICLES_PER_FEED, CACHE_DIR

logger = logging.getLogger(__name__)

# ── Simple file-based deduplication cache ────────────────────────────────────
_SEEN_URLS_FILE = CACHE_DIR / "seen_urls.txt"


def _load_seen_urls() -> set[str]:
    if _SEEN_URLS_FILE.exists():
        return set(_SEEN_URLS_FILE.read_text().splitlines())
    return set()


def _save_seen_url(url: str) -> None:
    with open(_SEEN_URLS_FILE, "a") as f:
        f.write(url + "\n")


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


# ── Article extraction ────────────────────────────────────────────────────────

def _extract_article_text(url: str, timeout: int = 10) -> tuple[str, str]:
    """
    Lightweight two-stage extraction:
    1. requests to fetch raw HTML
    2. BeautifulSoup to strip boilerplate
    Returns (clean_text, lead_paragraph)
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise nodes
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        # ET article body is usually inside <div class="artText"> or <div itemprop="articleBody">
        body = (
            soup.find("div", itemprop="articleBody")
            or soup.find("div", class_="artText")
            or soup.find("article")
            or soup.find("div", class_="article-body")
        )

        if body:
            paragraphs = [p.get_text(separator=" ", strip=True) for p in body.find_all("p")]
        else:
            # Fallback: grab all <p> tags
            paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]

        paragraphs = [p for p in paragraphs if len(p) > 60]
        full_text = " ".join(paragraphs)
        lead = paragraphs[0] if paragraphs else ""
        return full_text[:8000], lead   # Hard cap at 8 k chars

    except Exception as e:
        logger.warning("Article extraction failed for %s: %s", url, e)
        return "", ""


# ── Feed polling ──────────────────────────────────────────────────────────────

def fetch_new_articles(
    verticals: Optional[list[str]] = None,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Poll ET RSS feeds and return a list of article dicts.
    Each dict contains: id, url, title, summary, full_text, lead,
    published_at (ISO str), vertical, source.
    """
    seen_urls = set() if force_refresh else _load_seen_urls()
    feeds_to_poll = {
        k: v for k, v in ET_RSS_FEEDS.items()
        if verticals is None or k in verticals
    }
    articles: list[dict] = []

    for vertical, feed_url in feeds_to_poll.items():
        logger.info("Polling RSS: %s → %s", vertical, feed_url)
        try:
            feed = feedparser.parse(feed_url)
            entries = feed.entries[:MAX_ARTICLES_PER_FEED]
        except Exception as e:
            logger.error("RSS parse failed for %s: %s", vertical, e)
            continue

        for entry in entries:
            url = entry.get("link", "")
            if not url or url in seen_urls:
                continue

            # Parse publish timestamp
            try:
                pub_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                published_at = datetime(*pub_struct[:6], tzinfo=timezone.utc).isoformat() if pub_struct else datetime.now(timezone.utc).isoformat()
            except Exception:
                published_at = datetime.now(timezone.utc).isoformat()

            title   = entry.get("title", "").strip()
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(strip=True)[:500]

            # Full-text extraction
            full_text, lead = _extract_article_text(url)
            if not full_text:
                full_text = summary   # Fallback to RSS summary

            article = {
                "id":           _url_hash(url),
                "url":          url,
                "title":        title,
                "summary":      summary,
                "full_text":    full_text,
                "lead":         lead or summary[:200],
                "published_at": published_at,
                "vertical":     vertical,
                "source":       "Economic Times",
                "word_count":   len(full_text.split()),
            }
            articles.append(article)
            _save_seen_url(url)
            seen_urls.add(url)

            time.sleep(0.3)   # Be polite to the server

    logger.info("Fetched %d new articles from %d feeds.", len(articles), len(feeds_to_poll))
    return articles


def get_mock_articles() -> list[dict]:
    """
    Returns a set of high-quality mock articles for demo / offline usage.
    These allow the app to run without live internet access.
    """
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "id": "mock001",
            "url": "https://economictimes.indiatimes.com/mock/rbi-rate-cut",
            "title": "RBI Cuts Repo Rate by 25 bps to 6.25% — First Reduction in 4 Years",
            "summary": "The Reserve Bank of India's Monetary Policy Committee unanimously voted to reduce the benchmark repo rate by 25 basis points to 6.25%, the first rate cut since May 2020, citing easing inflation and slowing growth.",
            "full_text": """The Reserve Bank of India's Monetary Policy Committee (MPC) on Friday unanimously voted to cut the benchmark repo rate by 25 basis points to 6.25 per cent — the first reduction in nearly four years. The decision comes amid easing retail inflation, which fell to a 5-month low of 4.9% in November, and signs of slowing economic momentum, with GDP growth projected at 6.6% for FY2026. 
            
            RBI Governor Sanjay Malhotra, who chaired his first MPC meeting after taking charge in December, emphasised that the rate cut is aimed at supporting growth without compromising the inflation target of 4%. The Standing Deposit Facility (SDF) rate has been adjusted to 6.00%, and the Marginal Standing Facility (MSF) rate to 6.50%.
            
            Market reaction was immediate, with the Sensex surging 600 points and the Nifty 50 crossing 23,800 within minutes of the announcement. Bond yields fell sharply, with the 10-year benchmark yield dropping 12 basis points to 6.68%. Real estate and banking stocks led the rally, with HDFC Bank gaining 2.3% and SBI rising 3.1%.
            
            Economists at ICICI Securities noted that the rate cut cycle may extend to another 50-75 bps over the next 12 months if inflation remains anchored. However, global uncertainties — including US tariff policy and crude oil price volatility — could constrain the pace of easing. The Indian rupee held steady at 86.2 against the dollar following the announcement.""",
            "lead": "The Reserve Bank of India cut its benchmark repo rate by 25 basis points to 6.25%, the first reduction since May 2020.",
            "published_at": now,
            "vertical": "Economy",
            "source": "Economic Times",
            "word_count": 220,
        },
        {
            "id": "mock002",
            "url": "https://economictimes.indiatimes.com/mock/zepto-ipo",
            "title": "Zepto Files DRHP for ₹4,500 Cr IPO; Values Quick-Commerce Startup at $5 Billion",
            "summary": "Mumbai-based quick-commerce platform Zepto has filed its Draft Red Herring Prospectus with SEBI, seeking to raise ₹4,500 crore through a combination of fresh issue and offer-for-sale.",
            "full_text": """Mumbai-based quick-commerce platform Zepto has filed its Draft Red Herring Prospectus (DRHP) with the Securities and Exchange Board of India (SEBI), seeking to raise ₹4,500 crore through a combination of fresh issue (₹3,000 crore) and offer-for-sale (₹1,500 crore). The filing values the company at approximately $5 billion, making it one of the most anticipated IPOs of 2026.
            
            Founded in 2021 by IIT-dropout duo Aadit Palicha and Kaivalya Vohra, Zepto operates over 350 dark stores across 12 cities and processes more than 4 million orders daily. The company turned EBITDA-positive in Q2 FY2026, a milestone that has emboldened its public market ambitions.
            
            Key investors including Y Combinator, Nexus Venture Partners, Glade Brook Capital, and StepStone Group are among those likely to participate in the OFS. The IPO proceeds will be used for dark store expansion, technology upgrades, and working capital requirements.
            
            The move intensifies competition in the quick-commerce space, where Swiggy Instamart and Blinkit (owned by Zomato) already trade publicly. Analysts at Jefferies estimate the total addressable market for instant delivery in India could reach $40 billion by 2030. However, unit economics remain a concern — Zepto's average order value of ₹480 leaves thin margins in a category driven by discounts and logistics costs.
            
            SEBI is expected to review the DRHP within 30 days. Market conditions permitting, the IPO could launch in Q3 FY2026.""",
            "lead": "Quick-commerce startup Zepto files DRHP with SEBI for a ₹4,500 crore IPO, valuing the company at $5 billion.",
            "published_at": now,
            "vertical": "Startups",
            "source": "Economic Times",
            "word_count": 218,
        },
        {
            "id": "mock003",
            "url": "https://economictimes.indiatimes.com/mock/union-budget-2026",
            "title": "Union Budget 2026: FM Raises Income Tax Exemption Limit to ₹10 Lakh; Big Push for Green Energy",
            "summary": "Finance Minister Nirmala Sitharaman presented the Union Budget 2026-27, raising the personal income tax exemption limit to ₹10 lakh under the new regime and allocating ₹3.2 lakh crore for infrastructure.",
            "full_text": """Finance Minister Nirmala Sitharaman presented the Union Budget 2026-27 on February 1, raising the personal income tax exemption limit under the new regime from ₹7 lakh to ₹10 lakh — a move expected to benefit nearly 4.5 crore middle-class taxpayers and put an estimated ₹28,000 crore back into consumption.
            
            The budget's centrepiece is a ₹3.2 lakh crore capital expenditure outlay — a 15% increase over FY2025 — focused on roads, railways, and renewable energy. The government announced a National Green Hydrogen Mission allocation of ₹8,500 crore and production-linked incentives for domestic solar panel manufacturing worth ₹6,000 crore.
            
            On the fiscal front, the fiscal deficit target has been maintained at 4.4% of GDP for FY2027, demonstrating the government's commitment to consolidation despite the tax relief measures. Total expenditure is pegged at ₹50.65 lakh crore, while receipts (excluding borrowings) are estimated at ₹34.96 lakh crore.
            
            Agriculture received a major boost with a ₹2.5 lakh crore credit target for farmers and a new Kisan Credit Card scheme expansion. The startup ecosystem benefitted from the abolition of the Angel Tax and a reduced corporate tax rate of 22% for new manufacturing units established before March 2027.
            
            Bond markets reacted positively, with 10-year yields falling 8 bps. The Sensex gained 1,200 points intraday before paring gains to close 650 points higher. Rating agency Moody's called the budget "fiscally prudent with a growth orientation".""",
            "lead": "Finance Minister raised the income tax exemption to ₹10 lakh and allocated ₹3.2 lakh crore for infrastructure in the Union Budget 2026.",
            "published_at": now,
            "vertical": "Economy",
            "source": "Economic Times",
            "word_count": 245,
        },
        {
            "id": "mock004",
            "url": "https://economictimes.indiatimes.com/mock/tata-jlr-ev",
            "title": "Tata Motors Doubles Down on JLR EV Transition; Plans £2.5 Billion Investment in UK Gigafactory",
            "summary": "Tata Motors has announced a £2.5 billion commitment to build a battery gigafactory in Somerset, UK, to supply the fully electric Jaguar Land Rover lineup planned for launch by 2027.",
            "full_text": """Tata Motors has committed £2.5 billion to construct a state-of-the-art battery gigafactory in Bridgwater, Somerset — a project expected to create 4,000 direct jobs and supply lithium-ion cells for the fully electric Jaguar Land Rover (JLR) vehicle range.
            
            The announcement follows JLR's record quarterly profit of £1.02 billion in Q3 FY2026, driven by strong demand for its Range Rover and Defender models in North America and the Middle East. JLR's order book currently stands at 130,000 vehicles with an average wait time of 9 months.
            
            The gigafactory — with an initial capacity of 40 GWh, scalable to 80 GWh — will support JLR's goal of selling only battery-electric vehicles in key markets by 2030. The all-electric Jaguar XE concept showcased at Munich Motor Show received over 22,000 reservation deposits within 48 hours of unveiling.
            
            The UK government is co-investing £500 million through the Automotive Transformation Fund, recognising the strategic importance of keeping premium EV manufacturing onshore post-Brexit. Tata Group Chairman N. Chandrasekaran described the investment as "a defining moment for British manufacturing and Tata's long-term vision for sustainable mobility."
            
            Analysts at Morgan Stanley note that a successful JLR EV transition could lift Tata Motors' consolidated EBITDA margin by 300-400 basis points by FY2028. The stock rose 4.7% on the BSE following the announcement.""",
            "lead": "Tata Motors commits £2.5 billion to a UK battery gigafactory for Jaguar Land Rover's fully electric vehicle lineup.",
            "published_at": now,
            "vertical": "Auto",
            "source": "Economic Times",
            "word_count": 230,
        },
        {
            "id": "mock005",
            "url": "https://economictimes.indiatimes.com/mock/reliance-jio-ai",
            "title": "Reliance Jio Launches JioAI Cloud Platform; Targets 100 Million SME Users by 2027",
            "summary": "Reliance Jio unveiled JioAI Cloud, an affordable AI-as-a-service platform for Indian SMEs, offering large language model APIs, computer vision tools, and business automation at ₹999/month.",
            "full_text": """Reliance Jio officially launched JioAI Cloud at its annual technology summit in Mumbai, positioning the platform as India's most affordable enterprise AI solution aimed squarely at the country's 63 million small and medium enterprises.
            
            Priced starting at ₹999 per month, JioAI Cloud offers access to a suite of AI capabilities including a proprietary Hindi-English bilingual large language model (JioGPT-7B), computer vision APIs for inventory management and quality control, and pre-built workflow automation templates for GST filing, customer support, and demand forecasting.
            
            Akash Ambani, Chairman of Reliance Jio, said the platform is built to run efficiently on entry-level smartphones and 4G connections — critical for adoption in Tier 2 and Tier 3 cities. "We are not just democratising AI; we are Indianising it," he said, referencing the platform's deep vernacular support for 12 regional languages.
            
            The launch puts Jio in direct competition with Microsoft Azure AI, Google Vertex AI, and AWS Bedrock in India. However, analysts at CLSA note Jio's key differentiator is hyperlocal pricing and bundled Jio connectivity discounts that global players cannot match.
            
            JioAI Cloud already has 50,000 beta customers and has processed over 2 billion AI API calls during the trial period. Jio targets 10 million paid SME subscribers within 18 months and 100 million by 2027 — a target that would make it the largest AI services deployment in Asia by user count.""",
            "lead": "Reliance Jio launches JioAI Cloud at ₹999/month targeting India's 63 million SMEs with a bilingual AI platform.",
            "published_at": now,
            "vertical": "Tech",
            "source": "Economic Times",
            "word_count": 228,
        },
        {
            "id": "mock006",
            "url": "https://economictimes.indiatimes.com/mock/sebi-new-norms",
            "title": "SEBI Tightens F&O Margin Rules; Retail Participation in Derivatives Down 30% Since New Framework",
            "summary": "SEBI's enhanced margin requirements for futures and options, effective October 2025, have led to a 30% reduction in retail participation in equity derivatives while improving market stability metrics.",
            "full_text": """SEBI's enhanced margin framework for equity derivatives, which came into effect on October 1, 2025, has significantly reshaped India's futures and options market landscape — reducing retail participation by approximately 30% while improving key stability metrics, according to an internal SEBI review seen by the Economic Times.
            
            The new rules, which require upfront collection of true-to-label margins and eliminate the practice of netting client positions across brokers, were designed to curb speculative excess after a 2024 SEBI study found 93% of individual F&O traders lost money over a 3-year period.
            
            Total NSE F&O turnover has declined from a peak of ₹2,32,000 crore in daily notional value to approximately ₹1,60,000 crore — a 31% reduction. Retail brokerages including Zerodha, Groww, and Upstox have reported a 25-35% fall in active F&O clients, with some users migrating to direct equity investing or mutual funds.
            
            However, institutional participation has increased. FII activity in index futures has risen 22%, suggesting the market is attracting more sophisticated participants. Implied volatility on Nifty options has also declined — the India VIX has averaged 12.4 post-reform versus 15.8 pre-reform — indicating reduced systemic risk.
            
            The Association of Mutual Funds in India (AMFI) has reported a record ₹23,500 crore in SIP inflows for January 2026, suggesting some retail money previously in derivatives has rotated into mutual funds.""",
            "lead": "SEBI's new F&O margin rules have reduced retail derivatives participation by 30% while improving market stability.",
            "published_at": now,
            "vertical": "Markets",
            "source": "Economic Times",
            "word_count": 240,
        },
        {
            "id": "mock007",
            "url": "https://economictimes.indiatimes.com/mock/india-us-tariff",
            "title": "India and US Reach Preliminary Trade Deal; Tariff on IT Services Exports Capped at 2%",
            "summary": "India and the United States have concluded a preliminary bilateral trade agreement that caps tariffs on Indian IT services exports at 2% and grants preferential access for American agricultural products.",
            "full_text": """India and the United States have signed a preliminary bilateral trade agreement that resolves a long-standing dispute over digital services taxes and establishes a framework for reducing tariffs on goods and services traded between the two nations.
            
            The deal, announced jointly by Commerce Minister Piyush Goyal and US Trade Representative Katherine Tai in New Delhi, caps US tariffs on Indian IT and software service exports at 2%, significantly below the 10-25% tariffs threatened under the previous administration's digital services framework. India has in turn agreed to reduce import duties on American agricultural products including almonds, apples, and dairy — key demands from the US Farm Bureau.
            
            The IT sector — which exports over $220 billion annually to the US — welcomed the deal immediately. TCS, Infosys, and Wipro collectively gained over ₹45,000 crore in market capitalisation within hours of the announcement. NASSCOM called it "the single most important trade policy development for India's technology sector in a decade."
            
            The deal also includes a framework for an AI governance partnership, with both nations committing to joint standards development and reciprocal recognition of AI safety certifications — a provision that could accelerate the adoption of Indian-made AI products in the American market.
            
            However, the agreement still requires ratification by the US Congress and approval by India's Cabinet Committee on Economic Affairs. Opposition parties in India have raised concerns about the concessions made in agriculture, calling them "a capitulation to American farm lobby pressure".""",
            "lead": "India and the US sign a preliminary trade deal capping tariffs on Indian IT exports at 2% and granting US agricultural access.",
            "published_at": now,
            "vertical": "Economy",
            "source": "Economic Times",
            "word_count": 250,
        },
        {
            "id": "mock008",
            "url": "https://economictimes.indiatimes.com/mock/hdfc-bank-q3",
            "title": "HDFC Bank Q3 FY26 Net Profit Rises 18% to ₹16,736 Cr; NIM Expands to 3.7%",
            "summary": "HDFC Bank reported a net profit of ₹16,736 crore for Q3 FY2026, an 18% year-on-year increase, with the net interest margin expanding 20 basis points to 3.7% on the back of a favourable credit mix.",
            "full_text": """HDFC Bank, India's largest private sector lender, reported a net profit of ₹16,736 crore for the third quarter of FY2026, an 18.2% year-on-year increase, comfortably beating analyst estimates of ₹15,900 crore. The result was driven by robust loan growth, margin expansion, and moderation in credit costs.
            
            Net interest income (NII) grew 14.7% to ₹31,250 crore, while the net interest margin (NIM) improved 20 basis points sequentially to 3.7% — the highest in six quarters. The improvement reflects the bank's strategic pivot toward higher-yielding retail and SME loans, which now constitute 62% of the total loan book.
            
            Total advances grew 16.2% year-on-year to ₹26.2 lakh crore, with retail loans (home, auto, personal) growing fastest at 22% YoY. The CASA ratio held steady at 42.3%, indicating strong low-cost deposit franchise. Gross Non-Performing Assets (GNPA) ratio improved to 1.26% from 1.34% a year ago, reflecting disciplined underwriting.
            
            MD & CEO Sashidhar Jagdishan highlighted the successful integration of erstwhile HDFC Ltd. (merged in July 2023), noting that the mortgage book has now been fully re-priced to market rates. "The merger integration is behind us. We are now focused entirely on growth," he said.
            
            The stock rose 3.2% to ₹1,890 on the BSE. Analysts at Emkay Global raised the target price to ₹2,100, citing improving return ratios and a strengthening liability franchise.""",
            "lead": "HDFC Bank posts 18% profit growth in Q3 FY26, with NIM expanding to 3.7% driven by retail loan mix improvement.",
            "published_at": now,
            "vertical": "Markets",
            "source": "Economic Times",
            "word_count": 235,
        },
    ]
