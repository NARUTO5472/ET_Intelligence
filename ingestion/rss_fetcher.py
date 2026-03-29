"""
ingestion/rss_fetcher.py  (UPGRADED)
Fault-tolerant ET RSS ingestion with date-based smart fetching.

Key upgrades:
  • get_mock_articles() now returns 20 high-quality articles (was 8)
  • fetch_new_articles() accepts min_published filter for today-only ingestion
  • fetch_todays_articles() convenience wrapper for the sidebar button
"""

from __future__ import annotations
import hashlib
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup

from config import ET_RSS_FEEDS, MAX_ARTICLES_PER_FEED, CACHE_DIR

logger = logging.getLogger(__name__)

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


def _extract_article_text(url: str, timeout: int = 10) -> tuple[str, str]:
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
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "iframe", "noscript"]):
            tag.decompose()
        body = (
            soup.find("div", itemprop="articleBody")
            or soup.find("div", class_="artText")
            or soup.find("article")
            or soup.find("div", class_="article-body")
        )
        if body:
            paragraphs = [p.get_text(separator=" ", strip=True) for p in body.find_all("p")]
        else:
            paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        paragraphs = [p for p in paragraphs if len(p) > 60]
        full_text  = " ".join(paragraphs)
        lead       = paragraphs[0] if paragraphs else ""
        return full_text[:8000], lead
    except Exception as e:
        logger.warning("Article extraction failed for %s: %s", url, e)
        return "", ""


def fetch_new_articles(
    verticals: Optional[list[str]] = None,
    force_refresh: bool = False,
    min_published: Optional[datetime] = None,
) -> list[dict]:
    """
    Poll ET RSS feeds and return new article dicts.

    Args:
        verticals:     Limit to specific ET verticals (None = all).
        force_refresh: Ignore the seen-URL deduplication cache.
        min_published: Only return articles published on/after this datetime.
                       Pass datetime.now(utc).replace(hour=0,...) for today-only.
    """
    seen_urls   = set() if force_refresh else _load_seen_urls()
    feeds_to_poll = {
        k: v for k, v in ET_RSS_FEEDS.items()
        if verticals is None or k in verticals
    }
    articles: list[dict] = []

    for vertical, feed_url in feeds_to_poll.items():
        logger.info("Polling RSS: %s", vertical)
        try:
            feed    = feedparser.parse(feed_url)
            entries = feed.entries[:MAX_ARTICLES_PER_FEED]
        except Exception as e:
            logger.error("RSS parse failed for %s: %s", vertical, e)
            continue

        for entry in entries:
            url = entry.get("link", "")
            if not url or url in seen_urls:
                continue

            try:
                pub_struct   = entry.get("published_parsed") or entry.get("updated_parsed")
                published_at = (
                    datetime(*pub_struct[:6], tzinfo=timezone.utc)
                    if pub_struct else datetime.now(timezone.utc)
                )
            except Exception:
                published_at = datetime.now(timezone.utc)

            # ── Date filter: skip articles older than min_published ───────────
            if min_published and published_at < min_published:
                continue

            title   = entry.get("title", "").strip()
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(strip=True)[:500]
            full_text, lead = _extract_article_text(url)
            if not full_text:
                full_text = summary

            article = {
                "id":           _url_hash(url),
                "url":          url,
                "title":        title,
                "summary":      summary,
                "full_text":    full_text,
                "lead":         lead or summary[:200],
                "published_at": published_at.isoformat(),
                "vertical":     vertical,
                "source":       "Economic Times",
                "word_count":   len(full_text.split()),
            }
            articles.append(article)
            _save_seen_url(url)
            seen_urls.add(url)
            time.sleep(0.3)

    logger.info("Fetched %d new articles from %d feeds.", len(articles), len(feeds_to_poll))
    return articles


def fetch_todays_articles(verticals: Optional[list[str]] = None) -> list[dict]:
    """
    Convenience wrapper: fetches ONLY articles published today.
    Historical articles already in LanceDB are NOT re-fetched.
    This is what the sidebar 'Fetch Today's News' button calls.
    """
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    logger.info("Fetching articles published since %s.", today_start.isoformat())
    return fetch_new_articles(verticals=verticals, min_published=today_start)


# ── 20 high-quality mock articles for demo / offline mode ────────────────────

def get_mock_articles() -> list[dict]:
    """
    Returns 20 rich mock ET articles spanning all major verticals.
    Use for offline demos or when RSS is unavailable / geo-restricted.
    """
    now = datetime.now(timezone.utc).isoformat()
    yesterday = (datetime.now(timezone.utc) - timedelta(hours=18)).isoformat()

    return [
        # ── 1. Economy ────────────────────────────────────────────────────────
        {
            "id": "mock001", "url": "https://economictimes.indiatimes.com/mock/rbi-rate-cut",
            "title": "RBI Cuts Repo Rate by 25 bps to 6.25% — First Reduction in 4 Years",
            "summary": "The RBI MPC unanimously voted to cut the repo rate by 25 bps to 6.25%, the first cut since May 2020, citing easing inflation and slowing growth.",
            "full_text": """The Reserve Bank of India's Monetary Policy Committee (MPC) on Friday unanimously voted to cut the benchmark repo rate by 25 basis points to 6.25 per cent — the first reduction in nearly four years. The decision comes amid easing retail inflation, which fell to a 5-month low of 4.9% in November, and signs of slowing economic momentum, with GDP growth projected at 6.6% for FY2026.
RBI Governor Sanjay Malhotra, who chaired his first MPC meeting after taking charge in December, emphasised that the rate cut is aimed at supporting growth without compromising the inflation target of 4%. The SDF rate has been adjusted to 6.00%, and the MSF rate to 6.50%.
Market reaction was immediate, with the Sensex surging 600 points and the Nifty 50 crossing 23,800 within minutes of the announcement. Bond yields fell sharply, with the 10-year benchmark yield dropping 12 basis points to 6.68%. Real estate and banking stocks led the rally, with HDFC Bank gaining 2.3% and SBI rising 3.1%.
Economists at ICICI Securities noted that the rate cut cycle may extend to another 50-75 bps over the next 12 months if inflation remains anchored. However, global uncertainties — including US tariff policy and crude oil price volatility — could constrain the pace of easing. The Indian rupee held steady at 86.2 against the dollar following the announcement.""",
            "lead": "The Reserve Bank of India cut its benchmark repo rate by 25 bps to 6.25%, the first reduction since May 2020.",
            "published_at": now, "vertical": "Economy", "source": "Economic Times", "word_count": 220,
        },
        # ── 2. Startups ───────────────────────────────────────────────────────
        {
            "id": "mock002", "url": "https://economictimes.indiatimes.com/mock/zepto-ipo",
            "title": "Zepto Files DRHP for ₹4,500 Cr IPO; Values Quick-Commerce Startup at $5 Billion",
            "summary": "Quick-commerce platform Zepto filed its DRHP with SEBI, seeking to raise ₹4,500 crore through a fresh issue and OFS, valuing the company at $5 billion.",
            "full_text": """Mumbai-based quick-commerce platform Zepto has filed its Draft Red Herring Prospectus (DRHP) with SEBI, seeking to raise ₹4,500 crore through a combination of fresh issue (₹3,000 crore) and offer-for-sale (₹1,500 crore). The filing values the company at approximately $5 billion.
Founded in 2021 by IIT-dropout duo Aadit Palicha and Kaivalya Vohra, Zepto operates over 350 dark stores across 12 cities and processes more than 4 million orders daily. The company turned EBITDA-positive in Q2 FY2026.
Key investors including Y Combinator, Nexus Venture Partners, Glade Brook Capital, and StepStone Group are among those likely to participate in the OFS. IPO proceeds will be used for dark store expansion, technology upgrades, and working capital.
Analysts at Jefferies estimate the total addressable market for instant delivery in India could reach $40 billion by 2030. Unit economics remain a concern — Zepto's average order value of ₹480 leaves thin margins. SEBI is expected to review the DRHP within 30 days.""",
            "lead": "Quick-commerce startup Zepto files DRHP for ₹4,500 crore IPO at $5 billion valuation.",
            "published_at": now, "vertical": "Startups", "source": "Economic Times", "word_count": 185,
        },
        # ── 3. Economy ────────────────────────────────────────────────────────
        {
            "id": "mock003", "url": "https://economictimes.indiatimes.com/mock/union-budget-2026",
            "title": "Union Budget 2026: FM Raises Income Tax Exemption to ₹10 Lakh; ₹3.2 Lakh Cr Capex Push",
            "summary": "Finance Minister Nirmala Sitharaman raised the personal income tax exemption limit to ₹10 lakh and allocated ₹3.2 lakh crore for infrastructure in the Union Budget 2026-27.",
            "full_text": """Finance Minister Nirmala Sitharaman presented the Union Budget 2026-27 on February 1, raising the personal income tax exemption limit under the new regime from ₹7 lakh to ₹10 lakh — expected to benefit nearly 4.5 crore middle-class taxpayers and put ₹28,000 crore back into consumption.
The budget's centrepiece is a ₹3.2 lakh crore capital expenditure outlay — a 15% increase over FY2025 — focused on roads, railways, and renewable energy. The government announced a National Green Hydrogen Mission allocation of ₹8,500 crore and production-linked incentives for domestic solar panel manufacturing worth ₹6,000 crore.
The fiscal deficit target has been maintained at 4.4% of GDP for FY2027. Total expenditure is pegged at ₹50.65 lakh crore. Agriculture received a boost with a ₹2.5 lakh crore credit target and a new Kisan Credit Card scheme expansion. The startup ecosystem benefitted from abolition of the Angel Tax and a reduced corporate tax of 22% for new manufacturing units.
Bond markets reacted positively, with 10-year yields falling 8 bps. The Sensex gained 1,200 points intraday. Rating agency Moody's called it "fiscally prudent with a growth orientation".""",
            "lead": "FM raised income tax exemption to ₹10 lakh and allocated ₹3.2 lakh crore for infrastructure.",
            "published_at": now, "vertical": "Economy", "source": "Economic Times", "word_count": 210,
        },
        # ── 4. Auto ───────────────────────────────────────────────────────────
        {
            "id": "mock004", "url": "https://economictimes.indiatimes.com/mock/tata-jlr-ev",
            "title": "Tata Motors Doubles Down on JLR EV Transition; Plans £2.5 Billion UK Gigafactory",
            "summary": "Tata Motors committed £2.5 billion to a battery gigafactory in Somerset to supply the fully electric JLR vehicle lineup by 2027.",
            "full_text": """Tata Motors has committed £2.5 billion to construct a state-of-the-art battery gigafactory in Bridgwater, Somerset — expected to create 4,000 direct jobs and supply lithium-ion cells for the fully electric Jaguar Land Rover (JLR) vehicle range.
The announcement follows JLR's record quarterly profit of £1.02 billion in Q3 FY2026, driven by strong demand for Range Rover and Defender models in North America and the Middle East. JLR's order book currently stands at 130,000 vehicles with a 9-month average wait time.
The gigafactory — with an initial capacity of 40 GWh, scalable to 80 GWh — will support JLR's goal of selling only battery-electric vehicles in key markets by 2030. The UK government is co-investing £500 million through the Automotive Transformation Fund.
Analysts at Morgan Stanley note a successful EV transition could lift Tata Motors' consolidated EBITDA margin by 300-400 basis points by FY2028. The stock rose 4.7% on the BSE following the announcement.""",
            "lead": "Tata Motors commits £2.5 billion to a UK battery gigafactory for JLR's all-electric lineup.",
            "published_at": now, "vertical": "Auto", "source": "Economic Times", "word_count": 180,
        },
        # ── 5. Tech ───────────────────────────────────────────────────────────
        {
            "id": "mock005", "url": "https://economictimes.indiatimes.com/mock/reliance-jio-ai",
            "title": "Reliance Jio Launches JioAI Cloud Platform; Targets 100 Million SME Users by 2027",
            "summary": "Reliance Jio unveiled JioAI Cloud at ₹999/month, offering a bilingual LLM, computer vision APIs, and business automation tools for India's 63 million SMEs.",
            "full_text": """Reliance Jio officially launched JioAI Cloud at its annual technology summit in Mumbai, positioning the platform as India's most affordable enterprise AI solution for the country's 63 million SMEs.
Priced at ₹999 per month, JioAI Cloud offers access to JioGPT-7B (Hindi-English bilingual LLM), computer vision APIs for inventory management, and workflow automation templates for GST filing and customer support.
Akash Ambani said the platform runs efficiently on entry-level smartphones and 4G connections. The platform has deep vernacular support for 12 regional languages, differentiating it from AWS Bedrock, Google Vertex AI, and Microsoft Azure AI.
JioAI Cloud already has 50,000 beta customers and has processed over 2 billion AI API calls. Jio targets 10 million paid SME subscribers within 18 months and 100 million by 2027, which would make it the largest AI services deployment in Asia by user count.""",
            "lead": "Reliance Jio launches JioAI Cloud at ₹999/month targeting India's 63 million SMEs.",
            "published_at": now, "vertical": "Tech", "source": "Economic Times", "word_count": 165,
        },
        # ── 6. Markets ────────────────────────────────────────────────────────
        {
            "id": "mock006", "url": "https://economictimes.indiatimes.com/mock/sebi-new-norms",
            "title": "SEBI F&O Margin Rules Slash Retail Participation 30%; SIP Inflows Hit Record ₹23,500 Cr",
            "summary": "SEBI's enhanced margin framework for F&O, effective October 2025, has reduced retail derivatives participation by 30% while SIP inflows in mutual funds hit a record.",
            "full_text": """SEBI's enhanced margin framework for equity derivatives, effective October 1, 2025, has reduced retail participation by approximately 30% while improving key stability metrics. The new rules require upfront collection of true-to-label margins and eliminate netting of client positions across brokers.
Total NSE F&O turnover has declined from a peak of ₹2,32,000 crore in daily notional value to approximately ₹1,60,000 crore — a 31% reduction. Retail brokerages including Zerodha, Groww, and Upstox have reported a 25-35% fall in active F&O clients.
However, institutional participation has increased — FII activity in index futures has risen 22%. The India VIX has averaged 12.4 post-reform versus 15.8 pre-reform, indicating reduced systemic risk.
AMFI has reported a record ₹23,500 crore in SIP inflows for January 2026, suggesting retail money previously in derivatives has rotated into mutual funds — a structural positive for the industry.""",
            "lead": "SEBI's new F&O margin rules reduced retail derivatives participation by 30%, with SIP inflows reaching a record.",
            "published_at": now, "vertical": "Markets", "source": "Economic Times", "word_count": 185,
        },
        # ── 7. Economy ────────────────────────────────────────────────────────
        {
            "id": "mock007", "url": "https://economictimes.indiatimes.com/mock/india-us-tariff",
            "title": "India and US Reach Preliminary Trade Deal; IT Services Tariff Capped at 2%",
            "summary": "India and the US concluded a preliminary bilateral trade agreement capping tariffs on Indian IT exports at 2% and granting preferential US agricultural access.",
            "full_text": """India and the United States have signed a preliminary bilateral trade agreement that resolves a long-standing dispute over digital services taxes. The deal caps US tariffs on Indian IT and software service exports at 2% — well below the 10-25% tariffs previously threatened. India in turn agreed to reduce import duties on American agricultural products including almonds, apples, and dairy.
The IT sector — which exports over $220 billion annually to the US — welcomed the deal. TCS, Infosys, and Wipro collectively gained over ₹45,000 crore in market capitalisation within hours. NASSCOM called it "the single most important trade policy development for India's technology sector in a decade."
The deal includes a framework for an AI governance partnership with reciprocal recognition of AI safety certifications — a provision that could accelerate Indian-made AI products in the US market.
The agreement still requires ratification by the US Congress and CCEA approval. Opposition parties in India raised concerns about agricultural concessions.""",
            "lead": "India-US preliminary trade deal caps IT export tariffs at 2% and grants US agricultural market access.",
            "published_at": now, "vertical": "Economy", "source": "Economic Times", "word_count": 178,
        },
        # ── 8. Markets ────────────────────────────────────────────────────────
        {
            "id": "mock008", "url": "https://economictimes.indiatimes.com/mock/hdfc-bank-q3",
            "title": "HDFC Bank Q3 FY26 Net Profit Rises 18% to ₹16,736 Cr; NIM Expands to 3.7%",
            "summary": "HDFC Bank reported an 18% YoY rise in Q3 FY26 net profit to ₹16,736 crore, with NIM expanding 20 bps to 3.7% on a favourable credit mix.",
            "full_text": """HDFC Bank, India's largest private sector lender, reported a net profit of ₹16,736 crore for Q3 FY2026, an 18.2% year-on-year increase, beating analyst estimates of ₹15,900 crore. NII grew 14.7% to ₹31,250 crore, while NIM improved 20 bps sequentially to 3.7% — the highest in six quarters.
Total advances grew 16.2% YoY to ₹26.2 lakh crore, with retail loans growing fastest at 22% YoY. The CASA ratio held at 42.3%. Gross NPA ratio improved to 1.26% from 1.34% a year ago, reflecting disciplined underwriting.
MD & CEO Sashidhar Jagdishan highlighted successful integration of erstwhile HDFC Ltd. "The merger integration is behind us. We are now focused entirely on growth." The stock rose 3.2% to ₹1,890 on the BSE. Analysts at Emkay Global raised their target price to ₹2,100, citing improving return ratios.""",
            "lead": "HDFC Bank posts 18% profit growth in Q3 FY26 driven by NIM expansion and retail loan mix.",
            "published_at": now, "vertical": "Markets", "source": "Economic Times", "word_count": 165,
        },
        # ── 9. Startups ───────────────────────────────────────────────────────
        {
            "id": "mock009", "url": "https://economictimes.indiatimes.com/mock/ola-electric-q3",
            "title": "Ola Electric Q3 FY26 Revenue Up 68% to ₹1,890 Cr; Delivery Delays Persist in South India",
            "summary": "Ola Electric reported a 68% YoY revenue jump to ₹1,890 crore in Q3 FY26 but continued to face delivery delays in southern markets and mounting consumer complaints.",
            "full_text": """Ola Electric Mobility reported Q3 FY2026 revenue of ₹1,890 crore, up 68% year-on-year, driven by the S1 X and S1 Pro electric scooter models. However, the company's net loss widened to ₹495 crore as the company continues to invest heavily in manufacturing capacity at its Krishnapatnam Futurefactory.
The company dispatched 1.23 lakh scooters in the quarter but faced backlash over delivery timelines in Tamil Nadu, Karnataka, and Telangana, with consumer forums reporting an average delay of 6-8 weeks. Ola Electric's customer satisfaction score fell to 2.9 on Trustpilot versus rival Ather Energy's 4.3.
CEO Bhavish Aggarwal attributed delivery challenges to supply chain disruptions in battery cells and confirmed the company is sourcing cells from a second Tier-1 supplier to derisking. The company reiterated its target of 50 lakh annual capacity by FY2027 and remains the market leader in EV 2-wheelers with a 32% share.
Analysts at Motilal Oswal maintained a Buy rating with a revised target of ₹170, noting strong revenue growth despite operational challenges.""",
            "lead": "Ola Electric posts 68% revenue growth in Q3 FY26 but delivery delays in South India weigh on customer satisfaction.",
            "published_at": now, "vertical": "Auto", "source": "Economic Times", "word_count": 200,
        },
        # ── 10. Finance / Mutual Funds ────────────────────────────────────────
        {
            "id": "mock010", "url": "https://economictimes.indiatimes.com/mock/mf-nfo-boom",
            "title": "Mutual Fund NFO Boom: 12 New Schemes Raise ₹18,500 Cr in January; AMCs Eye Thematic Funds",
            "summary": "Asset management companies launched 12 new fund offerings in January 2026, collectively raising ₹18,500 crore, with thematic and sectoral funds dominating investor interest.",
            "full_text": """A record 12 new fund offerings (NFOs) launched in January 2026 collectively raised ₹18,500 crore, marking one of the busiest months for AMCs in recent history. Thematic funds — covering defence, infrastructure, and data centre REITs — attracted the lion's share of inflows.
HDFC AMC's Nifty Defence 50 Index Fund raised ₹4,200 crore in its three-week subscription window, the highest single-fund NFO collection since 2021. SBI Mutual Fund's Infrastructure Opportunities Fund and Nippon India's Data Economy Fund also exceeded their targets by 3x and 2x respectively.
However, financial planners caution investors against NFO euphoria. SEBI data shows that 68% of thematic NFOs from 2020-2023 underperformed their benchmark indices by more than 5 percentage points over a 3-year holding period.
Total AUM of the mutual fund industry crossed ₹68 lakh crore in January, with equity AUM at ₹39 lakh crore. SIP accounts reached 10 crore for the first time. AMFI chief N.S. Venkatesh warned that rising NFO activity should not distract investors from diversified equity and index funds for long-term wealth creation.""",
            "lead": "12 NFOs raised ₹18,500 crore in January 2026, with thematic funds dominating investor interest.",
            "published_at": now, "vertical": "Mutual Funds", "source": "Economic Times", "word_count": 210,
        },
        # ── 11. Tech ──────────────────────────────────────────────────────────
        {
            "id": "mock011", "url": "https://economictimes.indiatimes.com/mock/infosys-q3",
            "title": "Infosys Q3 FY26 Net Profit Up 14% to ₹7,021 Cr; Raises FY26 Revenue Guidance to 5.1-5.5%",
            "summary": "Infosys reported a 14% rise in Q3 FY26 net profit and raised its revenue growth guidance for FY2026, citing strong deal wins in AI transformation and cloud migration.",
            "full_text": """Infosys reported a net profit of ₹7,021 crore for Q3 FY2026, up 14% year-on-year, and raised its full-year revenue growth guidance to 5.1-5.5% in constant currency from 4.5-5%. Revenue for the quarter grew 8.9% YoY in USD terms to $4.93 billion.
The IT major won $2.6 billion in large deal total contract value (TCV) in Q3, the strongest quarter in two years. Key wins include a 7-year AI transformation partnership with a US healthcare insurer and a cloud migration project for a European financial services firm.
CEO Salil Parekh highlighted Infosys' AI-first strategy, noting that over 200 internal use cases have been deployed using its proprietary Infosys Topaz AI platform, delivering 18% productivity gains in software development.
The India headcount grew for the first time in three quarters, with 4,250 net additions. Utilisation improved to 83.2%. The stock rose 5.8% on BSE, its single-day biggest gain in 18 months. Analysts at CLSA raised the target to ₹2,050 citing improving deal wins and pricing power.""",
            "lead": "Infosys raises FY26 guidance after 14% profit growth, driven by strong AI and cloud deal wins.",
            "published_at": now, "vertical": "Tech", "source": "Economic Times", "word_count": 200,
        },
        # ── 12. Markets ───────────────────────────────────────────────────────
        {
            "id": "mock012", "url": "https://economictimes.indiatimes.com/mock/nifty-rally",
            "title": "Nifty 50 Crosses 24,000 on FII Return; Foreign Inflows Touch ₹32,000 Cr in March",
            "summary": "The Nifty 50 index crossed the 24,000 milestone as FIIs returned as net buyers, pumping ₹32,000 crore into Indian equities in March — the strongest monthly inflow since August 2023.",
            "full_text": """The Nifty 50 index crossed the 24,000 mark for the first time in six months on the back of a sharp reversal in foreign institutional investor (FII) flows. FIIs pumped ₹32,000 crore into Indian equities in March 2026, reversing five consecutive months of net selling totalling ₹1.2 lakh crore.
The return of FII buying has been attributed to three key factors: the US Federal Reserve's dovish pivot (pausing rate hikes with expectations of two cuts in 2026), India's improving GDP growth trajectory (projected at 7.2% for FY2027 by the IMF), and attractive valuations after the six-month correction.
Banking, IT, and FMCG stocks led the rally. Bajaj Finance gained 9.2%, ICICI Bank rose 7.8%, and TCS added 6.5% over the month. Small-cap indices have lagged, with the Nifty Smallcap 250 up only 2.3%.
SEBI chairperson Madhabi Puri Buch cautioned that the rally should be supported by earnings, not just liquidity. India's market cap-to-GDP ratio has returned to 1.1x, which she described as "warranting vigilance but not alarm".""",
            "lead": "Nifty 50 crosses 24,000 as FIIs return with ₹32,000 crore in March, reversing months of selling.",
            "published_at": now, "vertical": "Markets", "source": "Economic Times", "word_count": 205,
        },
        # ── 13. Energy ────────────────────────────────────────────────────────
        {
            "id": "mock013", "url": "https://economictimes.indiatimes.com/mock/adani-green",
            "title": "Adani Green Energy Commissions 2,000 MW Solar Plant in Rajasthan; Targets 50 GW by 2030",
            "summary": "Adani Green Energy commissioned its 2,000 MW Khavda solar plant in Rajasthan, making it one of the world's largest single-location renewable energy installations.",
            "full_text": """Adani Green Energy Limited (AGEL) has commissioned its 2,000 MW solar power plant at Khavda in the Rann of Kutch, Rajasthan — one of the world's largest single-location renewable energy installations. The plant will supply green power to 5 state electricity distribution companies across Gujarat, Rajasthan, Uttar Pradesh, Maharashtra, and Tamil Nadu under 25-year PPAs.
The Khavda project brings AGEL's total operational capacity to 12.3 GW, on track to meet its 2030 target of 50 GW. The plant was built in 22 months — 4 months ahead of schedule — with an investment of ₹12,000 crore.
Gautam Adani, chairman of the Adani Group, said the Khavda project demonstrates India's capacity to deliver renewable energy at scale. The plant uses domestically manufactured panels from Adani Solar, fulfilling the government's Atmanirbhar Bharat mandate.
AGEL stock rose 8.3% on the day of commissioning on BSE. Analysts at Morgan Stanley maintained an Overweight rating with a target of ₹1,850, noting AGEL's growing contracted revenue pipeline of ₹2.1 lakh crore and improving debt metrics.""",
            "lead": "Adani Green Energy commissions a 2,000 MW solar plant at Khavda, one of the world's largest single-location renewable facilities.",
            "published_at": yesterday, "vertical": "Energy", "source": "Economic Times", "word_count": 205,
        },
        # ── 14. Finance ───────────────────────────────────────────────────────
        {
            "id": "mock014", "url": "https://economictimes.indiatimes.com/mock/paytm-rbi",
            "title": "Paytm Payments Bank Gets Fresh RBI Nod for Restricted Operations; Stock Surges 22%",
            "summary": "The Reserve Bank of India granted Paytm Payments Bank a conditional approval to resume limited merchant payment services, ending a year-long regulatory freeze.",
            "full_text": """Paytm Payments Bank (PPBL) has received a conditional approval from the Reserve Bank of India to resume limited merchant payment services, ending a year-long regulatory freeze that had severely impacted One97 Communications' business. The approval allows PPBL to onboard new merchants and process UPI transactions, but stops short of permitting it to accept fresh consumer deposits.
The RBI cited improved compliance infrastructure, enhanced KYC systems, and satisfactory resolution of AML deficiencies as the basis for the partial reinstatement. Paytm must submit monthly compliance reports to the RBI for the next 12 months.
One97 Communications' stock surged 22% on the NSE, trading at ₹820 — though still significantly below its all-time high of ₹1,955. Vijay Shekhar Sharma, Paytm's founder, called the development "a new chapter of trust" and reiterated the company's commitment to full RBI compliance.
Analysts at Goldman Sachs estimate the partial reinstatement could restore approximately 65% of Paytm's pre-freeze merchant GMV within 6 months. The company has maintained its financial services businesses — lending, insurance distribution, and fastag — throughout the freeze period.""",
            "lead": "Paytm Payments Bank gets conditional RBI approval to resume merchant UPI services; stock gains 22%.",
            "published_at": yesterday, "vertical": "Finance", "source": "Economic Times", "word_count": 215,
        },
        # ── 15. Economy ───────────────────────────────────────────────────────
        {
            "id": "mock015", "url": "https://economictimes.indiatimes.com/mock/india-gdp-q3",
            "title": "India Q3 FY26 GDP Grows 6.8%; Consumption Recovery and Capex Spending Drive Beat",
            "summary": "India's Q3 FY2026 GDP expanded 6.8% YoY, beating the 6.4% consensus estimate, driven by strong private consumption and a 22% surge in government capital expenditure.",
            "full_text": """India's gross domestic product grew 6.8% year-on-year in the October-December quarter of FY2026 — beating the 6.4% consensus estimate — driven by a recovery in private consumption and a 22% surge in government capital expenditure. The data, released by the National Statistical Office (NSO), takes the full-year FY2026 GDP growth estimate to 6.6%.
Private final consumption expenditure grew 7.1% — the strongest print in five quarters — as rural consumption recovered on the back of a good kharif and rabi harvest. Urban consumption remained resilient despite moderation in formal sector hiring. Gross fixed capital formation grew 9.4%, with government capex leading at 22% YoY while private investment remained muted at 5.2%.
The manufacturing sector grew 6.9% and the services sector expanded 8.1%. Agriculture grew 4.2%, the best in three years, benefiting from above-normal monsoon rains and elevated MSPs.
Chief Economic Adviser V. Anantha Nageswaran attributed the beat to "synchronised fiscal and monetary policy support" and said India remains on track to be the fastest-growing major economy in the world for the fourth consecutive year. The IMF projects India's FY2027 growth at 7.0%.""",
            "lead": "India's Q3 FY26 GDP grew 6.8%, beating estimates, driven by consumption recovery and government capex.",
            "published_at": yesterday, "vertical": "Economy", "source": "Economic Times", "word_count": 220,
        },
        # ── 16. Markets ───────────────────────────────────────────────────────
        {
            "id": "mock016", "url": "https://economictimes.indiatimes.com/mock/bajaj-finance-nbfc",
            "title": "Bajaj Finance AUM Crosses ₹4 Lakh Crore; NBFC Sector Faces RBI Scrutiny on Pricing",
            "summary": "Bajaj Finance's total assets under management crossed ₹4 lakh crore as of Q3 FY26, while the RBI signalled tighter scrutiny on NBFC lending rates and fee structures.",
            "full_text": """Bajaj Finance's total assets under management (AUM) crossed ₹4 lakh crore as of Q3 FY2026 — a 30% YoY growth — cementing its position as India's largest NBFC by loan book. Customer franchise reached 9.8 crore, with new loans booked at 91 lakh during the quarter.
Net profit grew 18% to ₹4,308 crore. Asset quality remained pristine: gross NPA at 1.12% and net NPA at 0.48%. The company's B2B embedded finance business — where it co-lends with consumer electronics and two-wheeler dealers — saw the fastest growth at 42% YoY.
However, the Reserve Bank of India has flagged concerns about certain NBFC lending practices. An RBI circular issued last week requires all NBFCs with AUM above ₹1,000 crore to disclose all fees (processing, prepayment, penal) in a standardised Annual Percentage Rate (APR) format by April 2026. Bajaj Finance estimates the compliance will require UI changes across its app and physical application forms.
The stock was up 2.3% on the NSE. Analysts at JP Morgan maintained an Overweight with a target of ₹9,800, noting Bajaj Finance's superior risk-adjusted returns despite the regulatory headwind.""",
            "lead": "Bajaj Finance AUM crosses ₹4 lakh crore in Q3 FY26 as RBI tightens NBFC fee disclosure norms.",
            "published_at": yesterday, "vertical": "Finance", "source": "Economic Times", "word_count": 210,
        },
        # ── 17. Startups ──────────────────────────────────────────────────────
        {
            "id": "mock017", "url": "https://economictimes.indiatimes.com/mock/blinkit-vs-zepto",
            "title": "Quick Commerce War Intensifies: Blinkit Hits 1 Crore Daily Orders; Zepto IPO Pressure Mounts",
            "summary": "Zomato-owned Blinkit crossed 1 crore daily orders for the first time, threatening Zepto's growth narrative ahead of its planned IPO and reshaping the ₹60,000 crore quick commerce market.",
            "full_text": """Zomato-owned Blinkit crossed 1 crore daily orders for the first time in March 2026, a milestone that intensifies pressure on Zepto, which is currently in the process of filing its IPO. Blinkit's achievement comes just three years after Zomato acquired the erstwhile Grofers for ₹4,447 crore.
Blinkit now operates 1,000 dark stores across 47 cities, compared to Zepto's 350 stores in 12 cities. Average delivery time for Blinkit in metro areas is 9.2 minutes. Swiggy Instamart, the third major player, operates 750 dark stores but trails both in daily order volumes at 65 lakh.
The competition is compressing margins across the industry. Blinkit's take rate has fallen from 19% to 14% as it cuts commissions to attract restaurant partners. Zepto has responded by expanding into fresh produce and dairy, categories with higher basket sizes.
Analysts caution that the quick commerce market — estimated at ₹60,000 crore by 2027 — cannot sustain three profitable players at current burn rates. Consolidation is widely expected within 18 months, with Swiggy Instamart most frequently cited as a potential acquisition target.""",
            "lead": "Blinkit hits 1 crore daily orders, intensifying quick commerce competition as Zepto prepares its IPO.",
            "published_at": yesterday, "vertical": "Startups", "source": "Economic Times", "word_count": 215,
        },
        # ── 18. Healthcare ────────────────────────────────────────────────────
        {
            "id": "mock018", "url": "https://economictimes.indiatimes.com/mock/sun-pharma",
            "title": "Sun Pharma Eyes $2 Billion US Market Entry for Dermatology Biologic; FDA Fast-Track Granted",
            "summary": "Sun Pharmaceutical Industries received FDA Fast-Track designation for its novel dermatology biologic SPN-001, targeting the $2 billion US moderate-to-severe psoriasis market.",
            "full_text": """Sun Pharmaceutical Industries has received FDA Fast-Track designation for SPN-001, its proprietary IL-23 inhibitor biologic targeting moderate-to-severe psoriasis. The designation accelerates the pathway to US approval and could allow Sun Pharma to enter the $2 billion US dermatology biologics market as early as FY2028.
SPN-001 showed a 78% PASI-90 response rate in Phase 2 trials — comparable to Eli Lilly's Taltz (ixekizumab) and AbbVie's Skyrizi (risankizumab). Sun Pharma's management guides for Phase 3 initiation in Q1 FY2027 with 420 patients across 60 sites in the US, EU, and India.
The development is a strategic milestone for Sun Pharma's specialty business, which now contributes 38% of US revenue but has been primarily generic-driven. If approved, SPN-001 would be India's first globally competitive biologic in a high-value specialty indication.
MD Dilip Shanghvi said the Fast-Track designation validates the company's $800 million R&D spend over the past five years. The stock rose 6.2% on BSE. Analysts at UBS raised the target to ₹2,200, noting that SPN-001 alone could add ₹8,000-12,000 crore to Sun Pharma's valuation if approved.""",
            "lead": "Sun Pharma gets FDA Fast-Track designation for its psoriasis biologic, targeting the $2 billion US dermatology market.",
            "published_at": yesterday, "vertical": "Healthcare", "source": "Economic Times", "word_count": 215,
        },
        # ── 19. Politics / Economy ────────────────────────────────────────────
        {
            "id": "mock019", "url": "https://economictimes.indiatimes.com/mock/gst-revenue",
            "title": "GST Collections Hit ₹2.05 Lakh Crore in February 2026 — Highest Ever; FM Eyes 10% Full-Year Growth",
            "summary": "India's GST collections reached a record ₹2.05 lakh crore in February 2026, buoyed by robust consumption, improved compliance, and e-invoicing enforcement across MSME segments.",
            "full_text": """India's Goods and Services Tax (GST) collections hit a record ₹2.05 lakh crore in February 2026, surpassing the previous high of ₹1.87 lakh crore from April 2024. The surge was driven by robust domestic consumption, improved compliance enforcement, and the full rollout of e-invoicing for businesses with turnover above ₹5 crore.
CGST collections were ₹38,591 crore, SGST ₹47,935 crore, IGST ₹1,04,966 crore, and Cess ₹13,260 crore. The cumulative GST collection for April-February FY2026 stands at ₹20.2 lakh crore — 11.3% higher than the same period last year, putting the government on track to exceed its ₹22 lakh crore full-year budget target.
Finance Minister Nirmala Sitharaman attributed the buoyancy to improved taxpayer compliance (return filing rate now at 93%), enforcement of GST on online gaming platforms following the Supreme Court ruling, and a surge in FMCG and consumer electronics sales. The GST Council is likely to deliberate on rationalising the four-slab structure in its March meeting, potentially merging the 12% and 18% slabs.
India's tax-to-GDP ratio has improved to 12.1% in FY2026, the highest since Independence, according to revenue secretary Sanjay Malhotra.""",
            "lead": "GST collections hit a record ₹2.05 lakh crore in February 2026, driven by improved compliance and consumption.",
            "published_at": yesterday, "vertical": "Politics", "source": "Economic Times", "word_count": 225,
        },
        # ── 20. Mutual Funds ──────────────────────────────────────────────────
        {
            "id": "mock020", "url": "https://economictimes.indiatimes.com/mock/mutual-fund-index",
            "title": "Index Funds Overtake Active Large-Cap Funds in AUM for First Time; SIP Flows Tilt Passive",
            "summary": "For the first time, index funds and ETFs collectively surpassed active large-cap mutual funds in total AUM, as investors increasingly favour low-cost passive strategies.",
            "full_text": """Index funds and exchange-traded funds (ETFs) have collectively surpassed active large-cap mutual funds in AUM for the first time in Indian mutual fund history, according to AMFI data for February 2026. Passive large-cap AUM stands at ₹12.3 lakh crore versus active large-cap AUM of ₹11.9 lakh crore.
The tipping point has been driven by a structural shift in SIP inflows. Of ₹23,500 crore in monthly SIPs, approximately 31% now flows into passive funds — up from 18% two years ago. Nifty 50 and Nifty 500 index funds from UTI, HDFC AMC, and SBI MF dominate the inflow charts.
SEBI's data has repeatedly shown that over a 5-year period, 68% of active large-cap funds underperform the Nifty 50 TRI after accounting for expense ratios averaging 1.4% — compared to 0.05-0.10% for index funds. The evidence has steadily shifted retail investor behaviour.
However, active fund managers argue that mid-cap and small-cap categories still offer significant alpha opportunities that index funds cannot capture. HDFC AMC's Flexi-cap and Mirae Asset's Emerging Bluechip continue to outperform their benchmarks by 4-6% annually.
SEBI is reportedly considering reducing the maximum expense ratio cap for active funds from 2.25% to 1.75%, which would further accelerate the passive shift.""",
            "lead": "Index funds overtake active large-cap mutual funds in AUM for the first time as investors shift to passive strategies.",
            "published_at": yesterday, "vertical": "Mutual Funds", "source": "Economic Times", "word_count": 225,
        },
    ]
