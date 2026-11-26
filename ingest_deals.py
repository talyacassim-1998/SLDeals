import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from supabase import create_client

# Stream logs immediately in GitHub Actions
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODEL_NAME = "gemini-3.8-flash"
FORCE_SYNC = "--force" in sys.argv
REQUEST_TIMEOUT = (6, 12)  # (connect_timeout, read_timeout) in seconds

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

TARGET_URLS = {
    "mypromo_laugfs": "https://www.mypromo.lk/laugfssupermarket/promotions",
    "spar_offers_promotions": "https://spar2u.lk/collections/offers-promotions",
    "mypromo_arpico": "https://www.mypromo.lk/arpico/promotions",
    "mypromo_keells": "https://www.mypromo.lk/keellssuper/promotions",
    "lanka_sathosa_prices": "http://www.lankasathosa.org/prices",
    "glomark_deal_page": (
        "https://glomark.lk/deal-page"
        "?srsltid=AYCX9vKFAq1UH7w5ALa9iAhAyPEgD7fWHmLBuqm2Y6WQLh8tI3e8J59k"
    ),
}

BANK_OFFER_URLS = [
    ("ComBank", "https://www.combank.lk/rewards-promotions"),
    ("Sampath", "https://www.sampath.lk/sampath-cards/credit-card-offer?firstTab=super_markets"),
    ("BOC", "https://www.boc.lk/personal-banking/card-offers#supermarkets"),
    ("Seylan", "https://www.seylan.lk/promotions"),
    ("Pan Asia", "https://www.pabcbank.com/card-offers/"),
    ("American Express", "https://www.americanexpress.lk/en/offers/supermarket-offers"),
    ("Peoples Bank", "https://www.peoplesbank.lk/promotion-category/supermarkets/?cardType=credit_card"),
    ("Peoples Bank", "https://www.peoplesbank.lk/promotion-category/supermarkets/?cardType=debit_card"),
    ("Union Bank", "https://www.unionb.com/offer-category/supermarkets/"),
    ("NDB", "https://www.ndbbank.com/cards/card-offers/supermarkets"),
    ("Amana Bank", "https://www.amanabank.lk/personal/services/visa-debit-card/offers/supermarket.html"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

BANK_CODE_MAP = {
    "bank of ceylon": "BOC",
    "boc": "BOC",
    "commercial bank": "ComBank",
    "combank": "ComBank",
    "nations trust bank": "NTB",
    "ntb": "NTB",
    "american express": "American Express",
    "amex": "American Express",
    "dfcc bank": "DFCC",
    "dfcc": "DFCC",
    "seylan bank": "Seylan",
    "seylan": "Seylan",
    "people's bank": "Peoples Bank",
    "peoples bank": "Peoples Bank",
    "sampath bank": "Sampath",
    "sampath": "Sampath",
    "hatton national bank": "HNB",
    "hnb": "HNB",
    "pan asia": "Pan Asia",
    "pabc": "Pan Asia",
    "ndb": "NDB",
    "union bank": "Union Bank",
    "cargills bank": "Cargills Bank",
    "amana bank": "Amana Bank",
    "cdb": "CDB",
    "dialog pay": "Dialog Pay",
}

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def fetch_fast(url: str) -> requests.Response:
    """Fetches a URL with strict connect/read timeouts and 1 quick retry on 429."""
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 429:
        print(f"Rate limited (429) on {url}, waiting 4s...")
        time.sleep(4)
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


def clean_html_text(html: str, max_chars: int = 20000) -> str:
    """Strips scripts, styles, and nav noise from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "footer", "nav"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return text[:max_chars]


def extract_active_days(text: str) -> list[str]:
    """Maps offer text to standardized weekday array for the 7-day calendar & Checkout Mode."""
    lower = (text or "").lower()
    if "daily" in lower or "every day" in lower:
        return list(WEEKDAYS)
    return [d for d in WEEKDAYS if d.lower() in lower]


def ingest_official_bank_pages() -> None:
    """Continuously checks all 11 official Sri Lankan bank supermarket offer pages."""
    default_valid = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")

    for bank_name, url in BANK_OFFER_URLS:
        print(f"[{bank_name}] Checking {url}...")
        try:
            resp = fetch_fast(url)
        except Exception as exc:
            print(f"[{bank_name}] Skipping {url}: {exc}")
            continue

        text = clean_html_text(resp.text, max_chars=20000)
        if not text or len(text) < 120:
            print(f"[{bank_name}] Empty or JS-blocked content from {url}, skipping.")
            continue

        if not ai_client:
            print(f"[{bank_name}] Skipping LLM parse (GEMINI_API_KEY not set).")
            continue

        prompt = f"""
        Extract ONLY active Sri Lankan supermarket / grocery card offers from this {bank_name} webpage text.
        Ignore non-supermarket offers (hotels, dining, clothing, electronics).
        Return a JSON array of objects with these exact keys:
        - bank_name (use '{bank_name}')
        - card_type (string, e.g. '{bank_name} Credit Card', '{bank_name} Debit Card')
        - store_name (one of: 'Keells', 'Cargills', 'Arpico', 'Softlogic Glomark', 'Laugfs', 'SPAR', or 'All Participating Supermarkets')
        - discount_pct (integer or null)
        - applicable_categories (string, e.g. 'Total Bill', 'Fresh Vegetables, Fruits & Seafood')
        - active_day_of_week (string, e.g. 'Thursdays', 'Wednesdays', '18th–20th Dec', 'Daily')
        - active_days (array of weekday names from ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], or [] if specific calendar dates only)
        - min_bill_lkr (number or null)
        - max_discount_lkr (number or null)
        - exclusions (string summarizing excluded items like rice, milk powder, liquor, tobacco, gift vouchers)
        - is_online_only (boolean: true ONLY if valid strictly online like cargillsonline.com)
        - valid_until (YYYY-MM-DD string or null)

        TEXT:
        {text}
        """

        try:
            response = ai_client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            promos = json.loads(response.text)
        except Exception as exc:
            print(f"[{bank_name}] Gemini parse error on {url}: {exc}")
            continue

        rows = []
        for p in promos:
            if not p.get("store_name"):
                continue
            active_days = p.get("active_days") or extract_active_days(p.get("active_day_of_week", ""))
            rows.append(
                {
                    "bank_name": bank_name,
                    "card_type": p.get("card_type") or f"{bank_name} Card",
                    "store_name": p["store_name"],
                    "discount_pct": p.get("discount_pct"),
                    "applicable_categories": p.get("applicable_categories") or "Total Bill",
                    "active_day_of_week": p.get("active_day_of_week") or "Check Promo Terms",
                    "active_days": active_days,
                    "min_bill_lkr": p.get("min_bill_lkr"),
                    "max_discount_lkr": p.get("max_discount_lkr"),
                    "exclusions": p.get("exclusions") or "Excludes liquor, tobacco, gift vouchers & already-discounted items.",
                    "is_online_only": bool(p.get("is_online_only", False)),
                    "valid_until": p.get("valid_until") or default_valid,
                    "source_url": url,
                }
            )

        if rows:
            supabase.table("bank_promotions").delete().eq("source_url", url).execute()
            supabase.table("bank_promotions").insert(rows).execute()
            print(f"[{bank_name}] Synced {len(rows)} supermarket promos.")


def ingest_spar_offers_promotions() -> None:
    """Continuously checks https://spar2u.lk/collections/offers-promotions."""
    base_url = TARGET_URLS["spar_offers_promotions"]
    print(f"[SPAR] Checking {base_url}...")
    seen_handles = set()
    now_iso = datetime.now(timezone.utc).isoformat()
    valid_until = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
    rows = []

    for page_num in range(1, 3):
        json_url = f"{base_url}/products.json?limit=250&page={page_num}"
        if page_num > 1:
            time.sleep(2)
        try:
            resp = fetch_fast(json_url)
            products = resp.json().get("products", [])
        except Exception as exc:
            print(f"[SPAR] JSON page {page_num} warning: {exc}")
            products = []

        if not products:
            break

        for product in products:
            handle = product.get("handle")
            if not handle or handle in seen_handles or not product.get("variants"):
                continue

            variant = product["variants"][0]
            offer_price = float(variant.get("price") or 0)
            reg_price = float(variant.get("compare_at_price") or offer_price)
            if offer_price <= 0 or reg_price <= offer_price:
                continue

            seen_handles.add(handle)
            discount_pct = round(((reg_price - offer_price) / reg_price) * 100)
            rows.append(
                {
                    "store_name": "SPAR",
                    "branch_location": "All Island",
                    "raw_item_name": product["title"],
                    "canonical_item_name": product["title"],
                    "category": product.get("product_type") or "Pantry",
                    "regular_price": reg_price,
                    "offer_price": offer_price,
                    "discount_pct": discount_pct,
                    "valid_from": now_iso,
                    "valid_until": valid_until,
                    "source_url": f"https://spar2u.lk/products/{handle}",
                }
            )

    if rows:
        supabase.table("deals").delete().eq("store_name", "SPAR").execute()
        supabase.table("deals").insert(rows).execute()
        print(f"[SPAR] Synced {len(rows)} deals.")


def ingest_glomark_deal_page() -> None:
    """Continuously checks Softlogic Glomark deal-page."""
    url = TARGET_URLS["glomark_deal_page"]
    print(f"[Glomark] Checking {url}...")
    try:
        resp = fetch_fast(url)
    except Exception as exc:
        print(f"[Glomark] Skipping {url}: {exc}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    lines = [line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip()]
    now_iso = datetime.now(timezone.utc).isoformat()
    valid_until = (datetime.now(timezone.utc) + timedelta(days=6)).isoformat()
    seen_names = set()
    rows = []

    for i, line in enumerate(lines):
        pct = None
        offset = 0
        m1 = re.match(r"^(\d{1,2})%$", line)
        m2 = re.match(r"^(\d{1,2})%\s*Off$", line, re.IGNORECASE)
        if m1 and i + 1 < len(lines) and lines[i + 1].upper() == "OFF":
            pct = int(m1.group(1))
            offset = 2
        elif m2:
            pct = int(m2.group(1))
            offset = 1

        if not pct or i + offset + 3 >= len(lines):
            continue

        raw_line = lines[i + offset]
        item_name = re.sub(r"\s*-\s*in Sri Lanka$", "", raw_line, flags=re.IGNORECASE).strip()
        block = " ".join(lines[i + offset : min(i + offset + 8, len(lines))])
        if "OUT OF STOCK" in block.upper() or item_name in seen_names or len(item_name) < 3:
            continue

        prices = re.findall(r"Rs\.?\s*([0-9,]+(?:\.\d{2})?)", block)
        if not prices:
            continue

        offer_price = float(prices[0].replace(",", ""))
        reg_price = (
            float(prices[1].replace(",", ""))
            if len(prices) > 1
            else round(offer_price / (1 - pct / 100), 2)
        )
        seen_names.add(item_name)

        rows.append(
            {
                "store_name": "Softlogic Glomark",
                "branch_location": "All Island",
                "raw_item_name": item_name,
                "canonical_item_name": item_name,
                "category": "Pantry",
                "regular_price": reg_price,
                "offer_price": offer_price,
                "discount_pct": pct,
                "valid_from": now_iso,
                "valid_until": valid_until,
                "source_url": "https://glomark.lk/deal-page",
            }
        )

    if rows:
        supabase.table("deals").delete().eq("store_name", "Softlogic Glomark").execute()
        supabase.table("deals").insert(rows).execute()
        print(f"[Glomark] Synced {len(rows)} deals.")


def ingest_mypromo_store_pages() -> None:
    """Continuously checks MyPromo pages for LAUGFS, Arpico, and Keells."""
    mypromo_targets = [
        ("Laugfs", TARGET_URLS["mypromo_laugfs"]),
        ("Arpico", TARGET_URLS["mypromo_arpico"]),
        ("Keells", TARGET_URLS["mypromo_keells"]),
    ]
    default_valid = (datetime.now(timezone.utc) + timedelta(days=21)).strftime("%Y-%m-%d")
    bank_rows = []

    for store_name, url in mypromo_targets:
        print(f"[MyPromo-{store_name}] Checking {url}...")
        try:
            resp = fetch_fast(url)
        except Exception as exc:
            print(f"[MyPromo-{store_name}] Skipping {url}: {exc}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        lines = [line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip()]

        for i, line in enumerate(lines):
            headline_match = re.match(r"^(\d{1,2})%\s+(.+)$", line)
            if not headline_match or i + 4 >= len(lines):
                continue

            discount_pct = int(headline_match.group(1))
            title = headline_match.group(2).strip()
            window = " ".join(lines[i + 1 : min(i + 8, len(lines))])

            valid_until = default_valid
            end_match = re.search(r"Ends\s+(\d{2})/(\d{2})/(\d{4})", window)
            if end_match:
                valid_until = f"{end_match.group(3)}-{end_match.group(2)}-{end_match.group(1)}"

            matched_bank = None
            for key, code in BANK_CODE_MAP.items():
                if key in (title + " " + window).lower():
                    matched_bank = code
                    break

            if not matched_bank:
                continue

            min_match = re.search(r"Minimum bill(?:\s+over)?\s+Rs\.?\s*([0-9,\s]+)", window, re.IGNORECASE)
            max_match = re.search(r"maximum discount\s+Rs\.?\s*([0-9,\s]+)", window, re.IGNORECASE)
            min_bill = float(re.sub(r"[^\d]", "", min_match.group(1))) if min_match else None
            max_disc = float(re.sub(r"[^\d]", "", max_match.group(1))) if max_match else None
            active_days = extract_active_days(window)

            bank_rows.append(
                {
                    "bank_name": matched_bank,
                    "card_type": f"{matched_bank} Debit Card" if "debit" in title.lower() else f"{matched_bank} Credit Card",
                    "store_name": store_name,
                    "discount_pct": discount_pct,
                    "applicable_categories": "Fresh Vegetables, Fruits, Seafood & Meat" if "fresh" in window.lower() else "Total Bill",
                    "active_day_of_week": ", ".join(active_days) if active_days else "Check Promo Days",
                    "active_days": active_days,
                    "min_bill_lkr": min_bill,
                    "max_discount_lkr": max_disc,
                    "exclusions": "Excludes rice, milk powder, liquor, tobacco & gift vouchers.",
                    "is_online_only": False,
                    "valid_until": valid_until,
                    "source_url": url,
                }
            )

    if bank_rows:
        for _, url in mypromo_targets:
            supabase.table("bank_promotions").delete().eq("source_url", url).execute()
        supabase.table("bank_promotions").insert(bank_rows).execute()
        print(f"[MyPromo] Synced {len(bank_rows)} store-level card promotions.")


def ingest_sathosa_prices() -> None:
    """Checks Lanka Sathosa prices via RSS + official press releases."""
    print("[Sathosa] Checking RSS & price releases...")
    rss_url = (
        "https://news.google.com/rss/search"
        "?q=%22Lanka+Sathosa%22+(reduces+OR+price)+Sri+Lanka&hl=en-LK&gl=LK&ceid=LK:en"
    )
    fallback_articles = [
        "https://www.dailymirror.lk/breaking-news/Lanka-Sathosa-reduces-prices-of-six-essential-food-items/108-351178"
    ]
    article_urls = []

    try:
        rss_resp = fetch_fast(rss_url)
        root = ET.fromstring(rss_resp.content)
        for item in root.findall("./channel/item")[:3]:
            link = item.findtext("link") or ""
            if link:
                article_urls.append(link)
    except Exception:
        pass

    for url in fallback_articles:
        if url not in article_urls:
            article_urls.append(url)

    now_iso = datetime.now(timezone.utc).isoformat()
    valid_until = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    rows = []

    for link in article_urls:
        try:
            art_resp = requests.get(link, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if not art_resp.ok:
                continue
            text = clean_html_text(art_resp.text, max_chars=15000)
        except Exception:
            continue

        matches = re.findall(
            r"Rs\.?\s*(\d+(?:\.\d+)?)\s+for\s+a\s+kilogram(?:me)?\s+of\s+([a-zA-Z\s\(\)]+?)(?:,|and|\.|$)",
            text,
            re.IGNORECASE,
        )
        for price_str, item_raw in matches:
            clean_name = item_raw.strip().title()
            rows.append(
                {
                    "store_name": "Sathosa",
                    "branch_location": "All Island",
                    "raw_item_name": f"Lanka Sathosa {clean_name} (1kg)",
                    "canonical_item_name": f"{clean_name} 1kg",
                    "category": "Pantry",
                    "unit_size": "1kg",
                    "regular_price": None,
                    "offer_price": float(price_str),
                    "discount_pct": None,
                    "valid_from": now_iso,
                    "valid_until": valid_until,
                    "source_url": TARGET_URLS["lanka_sathosa_prices"],
                }
            )

    if rows:
        supabase.table("deals").delete().eq("store_name", "Sathosa").execute()
        supabase.table("deals").insert(rows).execute()
        print(f"[Sathosa] Synced {len(rows)} essential staple prices.")


if __name__ == "__main__":
    start = time.time()
    print(f"Starting sync at {datetime.now(timezone.utc).isoformat()}")
    ingest_spar_offers_promotions()
    ingest_glomark_deal_page()
    ingest_mypromo_store_pages()
    ingest_official_bank_pages()
    ingest_sathosa_prices()
    print(f"Sync complete in {time.time() - start:.1f}s.")
