import re
import logging
import urllib.parse
from html.parser import HTMLParser
from typing import Optional, Dict, Any, List
import httpx

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_VIDEO_BYTES = 50 * 1024 * 1024  # 50 MB

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Verified news domains (global & Indian news agencies, newspapers, public broadcasters, academic/gov)
VERIFIED_NEWS_DOMAINS = {
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "afp.com", "pti.in", "ptinews.com",
    "ndtv.com", "indiatoday.in", "thehindu.com", "indianexpress.com", "timesofindia.indiatimes.com",
    "hindustantimes.com", "news24online.com", "aajtak.in", "abplive.com", "zeenews.india.com",
    "livemint.com", "economictimes.indiatimes.com", "business-standard.com", "moneycontrol.com",
    "firstpost.com", "scroll.in", "theprint.in", "thewire.in", "newslaundry.com", "aninews.in",
    "ddnews.gov.in", "pib.gov.in", "india.gov.in", "nytimes.com", "theguardian.com",
    "washingtonpost.com", "wsj.com", "ft.com", "economist.com", "cnn.com", "aljazeera.com",
    "bloomberg.com", "cnbc.com", "forbes.com", "time.com", "newsweek.com", "usatoday.com",
    "npr.org", "pbs.org", "abcnews.go.com", "cbsnews.com", "nbcnews.com", "sky.com", "news.sky.com",
    "dw.com", "france24.com", "euronews.com", "abc.net.au", "cbc.ca", "japantimes.co.jp",
    "straitstimes.com", "scmp.com", "wikipedia.org", "en.wikipedia.org", "nature.com",
    "science.org", "scientificamerican.com", "nasa.gov", "isro.gov.in", "who.int", "un.org",
    "worldbank.org", "imf.org", "rbi.org.in", "espncricinfo.com", "espn.com", "cricbuzz.com",
    "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
}

# Verified broadcast news & educational channels on YouTube / video platforms
VERIFIED_VIDEO_CHANNELS = {
    "news 24", "news24", "ndtv", "ndtv india", "aaj tak", "india today", "abp news",
    "zee news", "times now", "wion", "firstpost", "ani news", "sansad tv", "dd news",
    "doordarshan", "the hindu", "the indian express", "hindustan times", "livemint",
    "economic times", "cnbc-tv18", "et now", "bbc news", "bbc", "reuters", "associated press",
    "cnn", "al jazeera english", "al jazeera", "sky news", "dw news", "france 24",
    "cnbc", "bloomberg television", "bloomberg", "abc news", "cbs news", "nbc news",
    "pbs newshour", "msnbc", "euronews", "cbc news", "abc news (australia)",
    "nasa", "isro official", "world health organization", "united nations", "ted",
    "national geographic", "veritasium", "smithsonian channel",
}


class _ArticleHTMLParser(HTMLParser):
    """Lightweight HTML parser to extract metadata, article text, and lead media from web pages."""

    IGNORE_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe"}

    def __init__(self):
        super().__init__()
        self._ignore_depth = 0
        self._current_tag: Optional[str] = None
        self.title_parts: List[str] = []
        self.h1_parts: List[str] = []
        self.paragraphs: List[str] = []
        self._current_p: List[str] = []
        self.meta: Dict[str, str] = {}
        self.images: List[str] = []
        self.videos: List[str] = []

    def handle_starttag(self, tag: str, attrs: list):
        tag_lower = tag.lower()
        if tag_lower in self.IGNORE_TAGS:
            self._ignore_depth += 1
            return

        attr_dict = {k.lower(): (v or "") for k, v in attrs if k}

        if tag_lower == "meta":
            prop = attr_dict.get("property", "").lower() or attr_dict.get("name", "").lower()
            content = attr_dict.get("content", "").strip()
            if prop and content:
                self.meta[prop] = content

        elif tag_lower == "img" and self._ignore_depth == 0:
            src = attr_dict.get("src") or attr_dict.get("data-src") or ""
            if src and not src.startswith("data:"):
                self.images.append(src)

        elif tag_lower in ("video", "source") and self._ignore_depth == 0:
            src = attr_dict.get("src") or ""
            if src and not src.startswith("data:"):
                self.videos.append(src)

        if self._ignore_depth == 0:
            self._current_tag = tag_lower
            if tag_lower in ("p", "article", "blockquote"):
                self._current_p = []

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in self.IGNORE_TAGS:
            if self._ignore_depth > 0:
                self._ignore_depth -= 1
            return

        if self._ignore_depth == 0:
            if tag_lower in ("p", "blockquote") and self._current_p:
                paragraph_text = " ".join("".join(self._current_p).split())
                if len(paragraph_text) >= 30:
                    self.paragraphs.append(paragraph_text)
                self._current_p = []
            if self._current_tag == tag_lower:
                self._current_tag = None

    def handle_data(self, data: str):
        if self._ignore_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        if self._current_tag == "title":
            self.title_parts.append(text)
        elif self._current_tag == "h1":
            self.h1_parts.append(text)
        elif self._current_p is not None and self._current_tag in ("p", "blockquote", "span", "a", "em", "strong", "b", "i"):
            self._current_p.append(" " + text + " ")


class URLIngestionService:
    """
    Fetches and extracts verifiable content (text, images, video streams, metadata,
    and publisher credibility signals) from news URLs, video platform URLs, and direct media links.
    """

    @staticmethod
    def normalize_url(raw_url: str) -> str:
        url = raw_url.strip()
        if not url:
            raise ValueError("Empty URL provided.")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc or "." not in parsed.netloc:
            raise ValueError(f"Invalid URL format: {raw_url}")
        return url

    @staticmethod
    def _is_verified_domain(domain: str) -> bool:
        d = domain.lower().lstrip("www.")
        if d in VERIFIED_NEWS_DOMAINS:
            return True
        for vd in VERIFIED_NEWS_DOMAINS:
            if d.endswith("." + vd):
                return True
        if d.endswith((".gov", ".gov.in", ".edu", ".ac.in", ".ac.uk", ".nic.in")):
            return True
        return False

    @staticmethod
    def _is_verified_channel(channel_name: str, title: str = "") -> bool:
        ch = (channel_name or "").lower().strip()
        if ch in VERIFIED_VIDEO_CHANNELS:
            return True
        for vc in VERIFIED_VIDEO_CHANNELS:
            if vc in ch:
                return True
        t_lower = (title or "").lower()
        for vc in ("news 24", "ndtv", "india today", "aaj tak", "bbc news", "reuters", "wion", "firstpost", "dd news", "the hindu"):
            if vc in t_lower:
                return True
        return False

    @staticmethod
    def _extract_youtube_id(url: str) -> Optional[str]:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower()
        if "youtu.be" in host:
            vid = parsed.path.lstrip("/").split("/")[0]
            return vid if vid else None
        if "youtube.com" in host:
            if parsed.path == "/watch":
                qs = urllib.parse.parse_qs(parsed.query)
                return qs.get("v", [None])[0]
            for prefix in ("/shorts/", "/embed/", "/v/"):
                if parsed.path.startswith(prefix):
                    return parsed.path[len(prefix):].split("/")[0]
        return None

    async def extract_from_url(self, raw_url: str) -> Dict[str, Any]:
        url = self.normalize_url(raw_url)
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        is_verified = self._is_verified_domain(domain)

        result: Dict[str, Any] = {
            "url": url,
            "domain": domain,
            "url_type": "news_article",
            "title": None,
            "publisher": domain,
            "is_verified_source": is_verified,
            "extracted_text": None,
            "image_bytes": None,
            "video_bytes": None,
            "media_url": None,
            "thumbnail_is_editorial": False,
        }

        async with httpx.AsyncClient(
            headers=BROWSER_HEADERS,
            follow_redirects=True,
            timeout=25.0,
            verify=False,
        ) as client:
            # 1. Check if YouTube URL for specialized metadata + thumbnail extraction
            yt_id = self._extract_youtube_id(url)
            if yt_id:
                result["url_type"] = "video_platform"
                await self._handle_youtube(client, url, yt_id, result)
                return result

            # 2. Fetch the URL
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.warning(f"Primary fetch failed for {url}: {e}")
                raise ValueError(f"Could not fetch URL ({url}): {str(e)}")

            # Update domain if redirected
            final_domain = urllib.parse.urlparse(str(resp.url)).netloc.lower()
            if final_domain.startswith("www."):
                final_domain = final_domain[4:]
            if final_domain:
                result["domain"] = final_domain
                result["is_verified_source"] = self._is_verified_domain(final_domain)

            content_type = resp.headers.get("content-type", "").lower()
            path_lower = parsed.path.lower()

            # 3. Direct Image URL
            if content_type.startswith("image/") or path_lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
                if len(resp.content) > MAX_IMAGE_BYTES:
                    raise ValueError("Image at URL exceeds maximum size limit (20 MB).")
                result["url_type"] = "image_url"
                result["title"] = parsed.path.split("/")[-1] or "Remote Image"
                result["image_bytes"] = resp.content
                result["media_url"] = str(resp.url)
                result["thumbnail_is_editorial"] = False
                return result

            # 4. Direct Video URL
            if content_type.startswith("video/") or path_lower.endswith((".mp4", ".mov", ".webm", ".mkv")):
                if len(resp.content) > MAX_VIDEO_BYTES:
                    raise ValueError("Video at URL exceeds maximum size limit (50 MB).")
                result["url_type"] = "video_url"
                result["title"] = parsed.path.split("/")[-1] or "Remote Video"
                result["video_bytes"] = resp.content
                result["media_url"] = str(resp.url)
                return result

            # 5. HTML Page (News Article or Video Page)
            html_text = resp.text
            parser = _ArticleHTMLParser()
            try:
                parser.feed(html_text)
            except Exception as e:
                logger.debug(f"HTML parsing warning for {url}: {e}")

            site_name = parser.meta.get("og:site_name") or final_domain
            result["publisher"] = site_name
            if not result["is_verified_source"] and self._is_verified_channel(site_name):
                result["is_verified_source"] = True

            # Extract Title
            title = (
                parser.meta.get("og:title")
                or parser.meta.get("twitter:title")
                or (" ".join(parser.h1_parts).strip() if parser.h1_parts else "")
                or (" ".join(parser.title_parts).strip() if parser.title_parts else "")
            )
            title = re.sub(r"\s+", " ", title).strip()

            # Extract Description
            description = (
                parser.meta.get("og:description")
                or parser.meta.get("description")
                or parser.meta.get("twitter:description")
                or ""
            )
            description = re.sub(r"\s+", " ", description).strip()

            # Combine article text
            body_text = "\n\n".join(parser.paragraphs[:40])
            text_segments = [seg for seg in (title, description, body_text) if seg]
            combined_text = "\n\n".join(text_segments).strip()

            # Fallback if page is JS-rendered and had minimal text: extract slug words
            if not combined_text:
                slug = parsed.path.strip("/").replace("-", " ").replace("_", " ").replace("/", " ")
                if len(slug) > 10:
                    combined_text = f"{site_name}: {slug}"

            result["title"] = title or f"Article from {final_domain}"
            result["extracted_text"] = combined_text if combined_text else None

            # Check for direct video in meta or <video> tags
            video_candidate = (
                parser.meta.get("og:video:secure_url")
                or parser.meta.get("og:video:url")
                or parser.meta.get("og:video")
                or (parser.videos[0] if parser.videos else None)
            )
            if video_candidate:
                resolved_vid = urllib.parse.urljoin(str(resp.url), video_candidate)
                if resolved_vid.lower().split("?")[0].endswith((".mp4", ".webm", ".mov")):
                    try:
                        v_resp = await client.get(resolved_vid, timeout=15.0)
                        if v_resp.status_code == 200 and v_resp.headers.get("content-type", "").startswith("video/"):
                            if len(v_resp.content) <= MAX_VIDEO_BYTES:
                                result["video_bytes"] = v_resp.content
                                result["media_url"] = resolved_vid
                                result["url_type"] = "video_platform"
                    except Exception as e:
                        logger.debug(f"Could not download embedded video {resolved_vid}: {e}")

            # Check for lead image (og:image / twitter:image / first article image)
            image_candidate = (
                parser.meta.get("og:image")
                or parser.meta.get("twitter:image")
                or (parser.images[0] if parser.images else None)
            )
            if image_candidate and not result["image_bytes"]:
                resolved_img = urllib.parse.urljoin(str(resp.url), image_candidate)
                try:
                    img_resp = await client.get(resolved_img, timeout=10.0)
                    if img_resp.status_code == 200 and len(img_resp.content) <= MAX_IMAGE_BYTES:
                        ct = img_resp.headers.get("content-type", "").lower()
                        if ct.startswith("image/") or resolved_img.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
                            result["image_bytes"] = img_resp.content
                            result["thumbnail_is_editorial"] = True
                            if not result["media_url"]:
                                result["media_url"] = resolved_img
                except Exception as e:
                    logger.debug(f"Could not download lead image {resolved_img}: {e}")

            if not result["extracted_text"] and not result["image_bytes"] and not result["video_bytes"]:
                raise ValueError(f"No verifiable text or media could be extracted from {url}.")

            return result

    async def _handle_youtube(
        self, client: httpx.AsyncClient, url: str, video_id: str, result: Dict[str, Any]
    ) -> None:
        title = ""
        author = ""
        description = ""

        # 1. Try YouTube oEmbed API (fast, reliable, no API key required)
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            oe_resp = await client.get(oembed_url, timeout=10.0)
            if oe_resp.status_code == 200:
                oe_data = oe_resp.json()
                title = oe_data.get("title", "")
                author = oe_data.get("author_name", "")
        except Exception as e:
            logger.debug(f"YouTube oEmbed failed: {e}")

        # 2. Fetch watch page HTML for description / meta tags
        try:
            page_resp = await client.get(f"https://www.youtube.com/watch?v={video_id}", timeout=12.0)
            if page_resp.status_code == 200:
                parser = _ArticleHTMLParser()
                parser.feed(page_resp.text)
                if not title:
                    title = parser.meta.get("og:title") or parser.meta.get("title") or ""
                description = parser.meta.get("og:description") or parser.meta.get("description") or ""
        except Exception as e:
            logger.debug(f"YouTube page fetch failed: {e}")

        # Check if channel is a verified news broadcaster
        is_verified_ch = self._is_verified_channel(author, title)
        result["publisher"] = author or "YouTube"
        result["is_verified_source"] = is_verified_ch

        # Clean hashtags and promotional boilerplate from YouTube description
        clean_desc = re.sub(r"#\w+", " ", description or "")
        clean_desc = re.sub(r"\s+", " ", clean_desc).strip()
        clean_title = re.sub(r"\|+", ", ", title or "").strip()

        result["title"] = title or f"YouTube Video ({video_id})"
        text_parts = [p for p in (clean_title, clean_desc) if p]
        result["extracted_text"] = ". ".join(text_parts).strip() or f"Video report from {author or 'YouTube'}"

        # 3. Fetch high-res video frame thumbnail for visual forensic inspection
        for thumb_quality in ("maxresdefault.jpg", "hqdefault.jpg", "mqdefault.jpg"):
            thumb_url = f"https://img.youtube.com/vi/{video_id}/{thumb_quality}"
            try:
                t_resp = await client.get(thumb_url, timeout=8.0)
                if t_resp.status_code == 200 and len(t_resp.content) > 1000:
                    result["image_bytes"] = t_resp.content
                    result["media_url"] = thumb_url
                    result["thumbnail_is_editorial"] = True
                    break
            except Exception:
                continue


url_service = URLIngestionService()
