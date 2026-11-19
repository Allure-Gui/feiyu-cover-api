"""音源解析：搜索歌曲 -> 拿到可下载的音频直链 -> 落地成文件。

当前默认使用 txqq 聚合接口（POST 表单，无需登录）。
接口结构参考社区点歌插件的实现；第三方接口可能失效，
届时把 config.source.provider 换成 "local" 或直接传 audio_url。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import urllib.parse

import httpx

logger = logging.getLogger("cover.source")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:146.0) "
    "Gecko/20100101 Firefox/146.0"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://music.txqq.pro",
    "Referer": "https://music.txqq.pro",
}


class SourceError(RuntimeError):
    """音源不可用或找不到歌曲。"""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


# ---------------------------------------------------------------- B 站音源
# 背景：像「雨爱 司凤版」这类是 B 站独有的 AI 二创 / 鬼畜 / 空耳视频，
# 根本不在 QQ、网易、酷狗的曲库里，音乐搜索只会返回同名伴奏。
# B 站的公开 API 可以直接拿到视频的纯音频流（DASH m4s，本质是 AAC），
# 正好是有完整人声的成品，适合拿来当翻唱输入。

# B 站风控对 UA 敏感：必须用完整 Chrome UA，Firefox UA + 假 buvid3 会在 playurl 触发 412。
_BILI_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
BILI_HEADERS = {
    "User-Agent": _BILI_UA,
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
_BILI_BUVID = "B1A2C3D4-E5F6-7890-ABCD-EF1234567890infoc"

# 命中这些词时优先走 B 站（这类二创/视频源音乐平台基本搜不到）
_BILI_HINTS = ("司凤", "鬼畜", "空耳", "二创", "b站", "bilibili", "B站",
               "邦吧", " DJ版", "phonk", "米塔", "公开发疯")


def _wants_bilibili(keyword: str) -> bool:
    low = keyword.lower()
    if re.search(r"\bBV[0-9A-Za-z]{10}\b", keyword):
        return True
    return any(h.lower() in low for h in _BILI_HINTS)


def _query_variants(keyword: str) -> list[str]:
    """接口对「歌名 + 歌手」的组合串支持很差，需要拆开逐个试。"""
    keyword = _clean(keyword)
    variants = [keyword]
    parts = [p for p in keyword.replace("　", " ").split(" ") if p]
    if len(parts) > 1:
        variants.append(parts[0])          # 歌名在前：取歌名
        variants.append(parts[-1])         # 歌手在前：取歌手
        variants.extend(parts)
    seen: list[str] = []
    for item in variants:
        if item and item not in seen:
            seen.append(item)
    return seen


_BAD_WORDS = ("伴奏", "翻唱", "纯音乐", "remix", "dj", "instrumental", "karaoke",
              "片段", "铃声", "童声", "儿歌", "钢琴", "吉他", "古筝", "二胡", "笛",
              "八音盒", "交响", "慢摇", "喊麦", "demo", "live", "现场", "演唱会",
              "女声版", "男声版", "合唱", "串烧", "抖音", "翻自", "cover", "纯享",
              "鬼畜", "拼好歌", "纯伴奏", "off vocal", "inst版")


def _version_tokens(keyword: str) -> list[str]:
    """抽出用户显式指定的版本标记，例如「司凤版」「女声版」「Live 版」。"""
    return [m.group(0) for m in re.finditer(r"[\u4e00-\u9fffA-Za-z0-9]{1,8}版", keyword)]


def _exempt_words(keyword: str) -> set[str]:
    """关键词里自己提到的"坏词"要豁免，否则用户点名要现场版反而被过滤掉。"""
    low = keyword.lower()
    asked = {w for w in _BAD_WORDS if w in low}
    if _version_tokens(keyword):
        # 用户明确要某个版本时，常见版本描述不再视为噪声
        asked |= {"翻唱", "cover", "女声版", "男声版", "live", "现场"}
    return asked


def _score(track: dict[str, Any], keyword: str, tokens: list[str],
           exempt: set[str] | None = None) -> int:
    """给候选打分：优先歌名精确、歌手匹配，排掉伴奏/儿歌/翻唱。"""
    title = track["title"].lower()
    artist = track["artist"].lower()
    key = keyword.lower()
    score = 0

    if title == key:
        score += 100
    elif title.startswith(key) or key in title:
        score += 45

    for token in tokens:
        low = token.lower()
        if low and low in artist:
            score += 55          # 歌手命中，权重最高
        if low and low in title:
            score += 10

    exempt = exempt or set()
    # 用户点名要的版本，命中就大幅加权
    for ver in _version_tokens(keyword):
        if ver.lower() in title:
            score += 120

    for bad in _BAD_WORDS:
        if bad in exempt:
            continue
        if bad in title or bad in artist:
            score -= 40

    if not track.get("url"):
        score -= 200
    return score


def _has_bad_word(track: dict[str, Any], exempt: set[str] | None = None) -> bool:
    text = (track["title"] + " " + track["artist"]).lower()
    exempt = exempt or set()
    return any(word in text for word in _BAD_WORDS if word not in exempt)


def _search_txqq(cfg: dict, keyword: str, limit: int = 5) -> list[dict[str, Any]]:
    """搜索歌曲，返回按相关度排序的 [{title, artist, url, cover, id}]。

    该聚合接口单页只回 4~5 条，且部分歌手（如周杰伦）只有翻唱版本，
    所以多翻一页并合并多个关键词变体，尽量凑够候选。
    """
    source = cfg.get("source", {})
    provider = (source.get("provider") or "txqq").lower()
    if provider != "txqq":
        raise SourceError(f"未支持的音源 provider: {provider}")

    base = source.get("txqq_base") or "https://music.txqq.pro/"
    headers = dict(HEADERS)
    if source.get("cookie"):
        headers["Cookie"] = source["cookie"]

    keyword = _clean(keyword)
    tokens = [t for t in keyword.split(" ") if t]
    collected: dict[str, dict[str, Any]] = {}

    with httpx.Client(timeout=20.0, headers=headers, follow_redirects=True) as client:
        for index, variant in enumerate(_query_variants(keyword)[:3]):
            pages = (1, 2) if index == 0 else (1,)
            for page in pages:
                form = {"input": variant, "filter": "name", "type": "qq", "page": page}
                try:
                    resp = client.post(base, data=form)
                    resp.raise_for_status()
                    payload = resp.json()
                except (httpx.HTTPError, ValueError):
                    continue
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list):
                    continue
                for item in data:
                    url = item.get("url") or item.get("link") or ""
                    if not url:
                        continue
                    track = {
                        "title": _clean(item.get("title") or ""),
                        "artist": _clean(item.get("author") or ""),
                        "url": url,
                        "cover": item.get("pic") or "",
                        "id": str(item.get("songid") or url[:32]),
                    }
                    collected.setdefault(track["id"], track)

    if not collected:
        raise SourceError(f"没搜到《{keyword}》，换首歌名试试")

    # 两级排序：先排除伴奏/翻唱/DJ/现场等版本，再按相关度
    ranked = sorted(
        collected.values(),
        key=lambda t: (_has_bad_word(t), -_score(t, keyword, tokens)),
    )
    ranked = [t for t in ranked if not _has_bad_word(t)] or ranked
    return ranked[:limit]


# ---------------------------------------------------------------- B 站音源实现

def _bili_client(cfg: dict) -> httpx.Client:
    """建一个带真实浏览器 UA 的会话（不在此处发请求）。

    注意：绝不能在这里调用 client.get()，否则 client 进入 open 状态，
    调用方再用 `with _bili_client() as client` 会抛
    "Cannot open a client instance more than once"。首页访问统一放到
    调用方的 with 块内（search / audio_url 各自已有）。
    用户若显式配了 bili_cookie 才写死 Cookie，否则让 httpx cookie jar
    接住首页下发的真实 buvid3（写死假值会导致 playurl 412）。
    """
    source = cfg.get("source", {})
    client = httpx.Client(timeout=20.0, headers=dict(BILI_HEADERS), follow_redirects=True)
    user_cookie = (source.get("bili_cookie") or "").strip()
    if user_cookie:
        client.headers["Cookie"] = user_cookie
    return client


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return _clean(text.replace("&quot;", '"').replace("&amp;", "&"))


def _bili_duration(value: Any) -> int:
    """B 站时长可能是秒数(140)，也可能是「2:20」「1:02:03」这种字符串。"""
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    parts = s.split(":")
    try:
        total = 0
        for p in parts:
            total = total * 60 + int(p)
        return total
    except ValueError:
        return 0


def _search_bilibili(cfg: dict, keyword: str, limit: int = 5) -> list[dict[str, Any]]:
    """B 站视频搜索。返回伪装的 track 结构，songmid 位置塞 bvid。"""
    source = cfg.get("source", {})
    api = source.get("bili_search_api") or "https://api.bilibili.com/x/web-interface/search/type"

    # 关键词清洗：去掉用户加的 B 站提示词，只留核心歌名
    q = keyword
    for h in _BILI_HINTS:
        q = q.replace(h.strip(), " ")
    q = _clean(q)

    with _bili_client(cfg) as client:
        try:
            client.get("https://www.bilibili.com/", timeout=8.0)
        except httpx.HTTPError:
            pass
        resp = client.get(api, params={"search_type": "video", "keyword": q, "page": 1},
                          headers={"Referer": "https://www.bilibili.com/"})
        resp.raise_for_status()
        payload = resp.json()
    if payload.get("code") != 0:
        logger.warning("B 站搜索失败：%s", payload.get("message"))
        return []

    results = ((payload.get("data") or {}).get("result") or [])
    # 命中用户指定版本词（如「司凤」「邦吧」）的排前面，避免原唱/无关视频混进来
    wanted = [h.strip().lower() for h in _BILI_HINTS if h.strip() and h.strip() in keyword.lower()]
    results.sort(key=lambda it: -sum(
        1 for w in wanted if w in _strip_html(it.get("title") or "").lower()))

    out: list[dict[str, Any]] = []
    for item in results:
        bvid = item.get("bvid") or ""
        if not bvid:
            continue
        out.append({
            "title": _strip_html(item.get("title") or ""),
            "artist": _clean(item.get("author") or ""),
            "bvid": bvid,
            "duration": _bili_duration(item.get("duration")),
            "url": "",
            "id": bvid,
            "origin": "bilibili",
        })
        if len(out) >= limit:
            break
    return out


def _bili_audio_url(cfg: dict, bvid: str) -> str:
    """bvid -> cid -> DASH 纯音频流直链（AAC，带人声）。

    B 站风控要求：view/playurl 必须在同一个会话里连续请求，且带真实 buvid3。
    412 时换新会话重试（风控是概率性的，换 cookie/UA 通常几次就过）。
    """
    source = cfg.get("source", {})
    view_api = source.get("bili_view_api") or "https://api.bilibili.com/x/web-interface/view"
    play_api = source.get("bili_play_api") or "https://api.bilibili.com/x/player/playurl"
    referer = f"https://www.bilibili.com/video/{bvid}"

    for attempt in range(3):
        try:
            with _bili_client(cfg) as client:
                # 再次访问视频页，刷新 cookie 并带上正确 Referer（缓解 412）
                try:
                    client.get(referer, timeout=8.0)
                except httpx.HTTPError:
                    pass

                r1 = client.get(view_api, params={"bvid": bvid}, headers={"Referer": referer})
                r1.raise_for_status()
                view = r1.json()
                if view.get("code") != 0:
                    return ""
                cid = (view.get("data") or {}).get("cid")
                if not cid:
                    return ""

                r2 = client.get(play_api, params={
                    "bvid": bvid, "cid": cid, "qn": 64, "fnval": 16, "fourk": 0,
                }, headers={"Referer": referer})
                if r2.status_code == 412:
                    logger.warning("B 站 playurl 412（第 %d 次），重试", attempt + 1)
                    continue
                r2.raise_for_status()
                play = r2.json()
            if play.get("code") != 0:
                logger.warning("B 站 playurl 返回 code=%s：%s", play.get("code"), play.get("message"))
                continue

            data = play.get("data") or {}
            # DASH 的 audio 是纯音频流（无画面，省带宽）；按码率从高到低挑
            audios = (data.get("dash") or {}).get("audio") or []
            if audios:
                best = max(audios, key=lambda a: int(a.get("bandwidth") or 0))
                return best.get("baseUrl") or best.get("base_url") or ""
            # 少数老视频只有 durl（音视频合流），用它兜底
            durl = data.get("durl") or []
            if durl:
                return durl[0].get("url") or ""
            return ""
        except httpx.HTTPError as exc:
            logger.warning("B 站取址失败（第 %d 次）：%s", attempt + 1, exc)
            continue
    return ""


# ---------------------------------------------------------------- QQ 官方接口

def _qq_cookie(cfg: dict) -> str:
    """读取 cookie：优先配置里的字符串，其次读文件。"""
    source = cfg.get("source", {})
    raw = (source.get("qq_cookie") or "").strip()
    if raw:
        return raw
    path = source.get("qq_cookie_file") or "/mnt/g/feiyu-cover/qq-cookie.txt"
    f = Path(path).expanduser()
    if not f.is_file():
        return ""
    text = f.read_text(encoding="utf-8", errors="replace")
    pairs = dict(re.findall(r"([A-Za-z_][\w]*)\s*[:=]\s*([^;\s]+)", text))
    return "; ".join(f"{k}={v}" for k, v in pairs.items())


def _qq_uin(cookie: str) -> str:
    m = re.search(r"(?:^|;\s*)uin=o?(\d+)", cookie)
    return m.group(1) if m else "0"


def _search_qq(cfg: dict, keyword: str, limit: int = 5) -> list[dict[str, Any]]:
    """QQ 音乐官方搜索，比第三方聚合接口准得多（能拿到 songmid / 专辑 / 时长）。"""
    cookie = _qq_cookie(cfg)
    headers = {"User-Agent": HEADERS.get("User-Agent", ""), "Referer": "https://y.qq.com/"}
    if cookie:
        headers["Cookie"] = cookie

    resp = httpx.get(
        "https://c.y.qq.com/soso/fcgi-bin/client_search_cp",
        params={"w": keyword, "format": "json", "n": max(limit, 10), "p": 1,
                "cr": 1, "g_tk": 5381, "t": 0},
        headers=headers, timeout=15.0, follow_redirects=True,
    )
    resp.raise_for_status()
    text = resp.text.strip()
    if text.startswith("callback"):
        text = text[text.index("(") + 1: text.rindex(")")]
    payload = json.loads(text)

    songs = (payload.get("data") or {}).get("song", {}).get("list") or []
    out: list[dict[str, Any]] = []
    for item in songs:
        mid = (item.get("songmid") or "").strip()
        name = _clean(item.get("songname") or "")
        if not mid or not name:
            continue
        singers = item.get("singer") or []
        artist = _clean(",".join(x.get("name", "") for x in singers if isinstance(x, dict)))
        out.append({
            "title": name,
            "artist": artist,
            "album": _clean(item.get("albumname") or ""),
            "duration": int(item.get("interval") or 0),
            "songmid": mid,
            "url": "",
            "cover": "",
            "id": mid,
            "origin": "qq",
        })
    return out


def _qq_audio_url(cfg: dict, songmid: str) -> str:
    """用 cookie 换完整地址；普通 VIP 只能拿 128k（M500）。"""
    cookie = _qq_cookie(cfg)
    if not cookie:
        return ""
    uin = _qq_uin(cookie)
    payload = {
        "req_0": {"module": "vkey.GetVkeyServer", "method": "CgiGetVkey", "param": {
            "guid": "10000", "songmid": [songmid], "songtype": [0], "uin": uin,
            "loginflag": 1, "platform": "20",
            "filename": [f"M500{songmid}.mp3"],
        }},
        "comm": {"uin": uin, "format": "json", "ct": 24, "cv": 0},
    }
    url = ("https://u.y.qq.com/cgi-bin/musicu.fcg?format=json&data="
           + urllib.parse.quote(json.dumps(payload)))
    resp = httpx.get(url, headers={"Cookie": cookie, "Referer": "https://y.qq.com/",
                                   "User-Agent": HEADERS.get("User-Agent", "")}, timeout=15.0)
    resp.raise_for_status()
    info = ((resp.json().get("req_0") or {}).get("data") or {}).get("midurlinfo") or [{}]
    purl = (info[0].get("purl") or "").strip()
    return ("https://ws.stream.qqmusic.qq.com/" + purl) if purl else ""


def _norm_name(text: str) -> str:
    """去掉空格和常见分隔符，便于粗略比对歌手名。"""
    return re.sub(r"[\s,，、/&·]+", "", (text or "")).lower()


def _char_sim(a: str, b: str) -> float:
    """字符重合度。用来绕过繁简差异（周杰倫 vs 周杰伦、葉惠美 vs 叶惠美）。"""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa), len(sb))


def _name_match(expected: str, got: str, threshold: float = 0.6) -> bool:
    """名字比对：先看互相包含，再看字符重合度（阈值 0.6 能跨过繁简）。"""
    exp, g = _norm_name(expected), _norm_name(got)
    if not exp or not g:
        return False
    if exp == g or exp in g or g in exp:
        return True
    return _char_sim(exp, g) >= threshold


def _artist_match(expected: str, candidates: Any) -> bool:
    """判断返回的歌手里有没有"同一个"——防止把同名翻唱当成原唱。"""
    exp = (expected or "").split(",")[0]
    if not _norm_name(exp):
        return False
    if isinstance(candidates, str):
        candidates = [candidates]
    return any(_name_match(exp, str(name)) for name in (candidates or []))


def _album_match(expected: str, got: str) -> bool:
    """专辑校验：翻唱通常用自己的单曲名当专辑，原唱才对得上专辑名。"""
    exp = _norm_name(expected)
    if not exp:
        return True          # 没有期望专辑就不校验
    if not _norm_name(got):
        return False
    return _name_match(expected, got)


def _gd_pick(cfg: dict, source: str, title: str, artist: str,
             album: str = "") -> tuple[str, str]:
    """在指定音源里搜索并**校验歌手**，返回 (url, 曲名)。

    网易云没有周杰伦的版权，只搜歌名会拿到「晴天(原唱周杰伦)-某个翻唱歌手」，
    所以必须校验歌手，否则会把翻唱当成原曲。
    """
    api = cfg.get("source", {}).get("gd_api") or "https://music-api.gdstudio.xyz/api.php"
    name = f"{title} {artist.split(',')[0]}".strip() if artist else title
    try:
        r = httpx.get(api, params={"types": "search", "source": source,
                                   "name": name, "count": 10, "pages": 1}, timeout=15.0)
        hits = r.json()
    except (httpx.HTTPError, ValueError):
        return "", ""
    if not isinstance(hits, list) or not hits:
        return "", ""

    # 歌手 + 专辑双重校验：只看歌手会被"标成原唱歌手"的翻唱骗过去
    matched = [
        h for h in hits
        if _artist_match(artist, h.get("artist")) and _album_match(album, str(h.get("album") or ""))
    ]
    if not matched:
        logger.info("音源 %s 没有《%s》的原唱版本（歌手/专辑对不上，视为翻唱）", source, title)
        return "", ""

    song_id = matched[0].get("id")
    if not song_id:
        return "", ""
    try:
        r2 = httpx.get(api, params={"types": "url", "source": source,
                                    "id": song_id, "br": 320}, timeout=15.0)
        data = r2.json()
    except (httpx.HTTPError, ValueError):
        return "", ""
    url = (data.get("url") or "") if isinstance(data, dict) else ""
    return (url, source) if url else ("", "")


def _gd_lenient(cfg: dict, title: str, artist: str) -> tuple[str, str]:
    """严格校验失败后的兜底：直接取搜索首条（通常是翻唱），也算能唱。"""
    api = cfg.get("source", {}).get("gd_api") or "https://music-api.gdstudio.xyz/api.php"
    sources = cfg.get("source", {}).get("gd_sources") or ["netease", "joox"]
    q = f"{title} {artist.split(',')[0]}".strip() if artist else title
    for source in sources:
        try:
            r = httpx.get(api, params={"types": "search", "source": source,
                                       "name": q, "count": 5, "pages": 1}, timeout=15.0)
            hits = r.json()
            if not isinstance(hits, list) or not hits:
                continue
            song_id = hits[0].get("id")
            if not song_id:
                continue
            r2 = httpx.get(api, params={"types": "url", "source": source,
                                        "id": song_id, "br": 320}, timeout=15.0)
            data = r2.json()
            url = (data.get("url") or "") if isinstance(data, dict) else ""
            if url:
                logger.info("音源 %s 只有翻唱版本，按兜底使用：%s", source, hits[0].get("name"))
                return url, source
        except (httpx.HTTPError, ValueError):
            continue
    return "", ""


def _gd_audio_url(cfg: dict, title: str, artist: str, album: str = "") -> tuple[str, str, bool]:
    """优先找原唱；找不到就退回翻唱（用户要求：宁可翻唱也不要失败）。

    返回 (url, 音源, 是否为翻唱兜底)。
    """
    sources = cfg.get("source", {}).get("gd_sources") or ["netease", "joox"]
    for source in sources:
        url, _ = _gd_pick(cfg, source, title, artist, album)
        if url:
            logger.info("音源 %s 命中《%s》原唱", source, title)
            return url, source, False

    if not cfg.get("source", {}).get("allow_cover_fallback", True):
        return "", "", False

    url, source = _gd_lenient(cfg, title, artist)
    return url, source, bool(url)


def best_url(cfg: dict, track: dict[str, Any]) -> tuple[str, str, bool]:
    """按质量依次尝试，返回 (url, 来源标记, 是否翻唱兜底)。"""
    source = cfg.get("source", {})
    order = source.get("order") or ["qq", "gd", "txqq"]
    mid = track.get("songmid") or ""

    # B 站 track 必须走 B 站取址：它带的是 bvid 而非 songmid，
    # 落到 gd/txqq 只会被当成普通歌名搜到同名伴奏（这正是司凤版的老问题）。
    # 强制提前处理，不依赖用户配置 order 里是否写了 bilibili。
    if track.get("bvid"):
        try:
            u = _bili_audio_url(cfg, track["bvid"])
            if u:
                return u, "bilibili", False
        except Exception as exc:  # noqa: BLE001
            logger.warning("B 站取址失败：%s", exc)

    for kind in order:
        try:
            if kind == "qq" and mid:
                u = _qq_audio_url(cfg, mid)
                if u:
                    return u, "qq", False
            elif kind == "gd":
                u, src, cover = _gd_audio_url(cfg, track.get("title", ""),
                                              track.get("artist", ""), track.get("album", ""))
                if u:
                    return u, src or "netease", cover
            elif kind == "txqq" and track.get("url"):
                return track["url"], "txqq", False
        except Exception as exc:  # noqa: BLE001
            logger.warning("音源 %s 取址失败：%s", kind, exc)
    return "", "", False


def search(cfg: dict, keyword: str, limit: int = 5) -> list[dict[str, Any]]:
    """优先用 QQ 官方搜索，补不足再走第三方聚合。

    B 站二创（司凤版/鬼畜/空耳等）在音乐平台搜不到，命中特征词时改走 B 站。
    """
    keyword = _clean(keyword)
    tokens = [t for t in keyword.split(" ") if t]

    # B 站专属二创：音乐平台拿到的只会是同名伴奏，直接走 B 站
    source_cfg = cfg.get("source", {})
    if _wants_bilibili(keyword) and source_cfg.get("bili_enabled", True):
        try:
            bili = _search_bilibili(cfg, keyword, limit)
            if bili:
                logger.info("B 站命中《%s》%d 条视频", keyword, len(bili))
                return bili
        except Exception as exc:  # noqa: BLE001
            logger.warning("B 站搜索失败，回退音乐平台：%s", exc)

    tracks: list[dict[str, Any]] = []
    try:
        tracks = _search_qq(cfg, keyword, limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("QQ 搜索失败，改用聚合接口：%s", exc)

    if len(tracks) < limit:
        try:
            tracks += _search_txqq(cfg, keyword, limit)
        except SourceError:
            pass

    if not tracks:
        raise SourceError(f"没搜到《{keyword}》，换首歌名试试")

    # 去重（同名同歌手只留一条），再按"是否伴奏/DJ"和相关度排序
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for t in tracks:
        key = (t.get("title", ""), t.get("artist", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)

    exempt = _exempt_words(keyword)
    ranked = sorted(
        unique,
        key=lambda t: (_has_bad_word(t, exempt), -_score(t, keyword, tokens, exempt)),
    )
    kept = [t for t in ranked if not _has_bad_word(t, exempt)]
    return (kept or ranked)[:limit]


def download(url: str, dest: Path, cfg: dict) -> Path:
    """下载音频到 dest（自动补扩展名）。返回实际路径。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": UA, "Referer": "https://music.txqq.pro/"}
    # B 站 CDN 校验 Referer，必须是 bilibili.com 才能下载
    if "bilibili" in url or "bilivideo" in url:
        headers["Referer"] = "https://www.bilibili.com/"
        headers["Cookie"] = cfg.get("source", {}).get("bili_cookie") or f"buvid3={_BILI_BUVID};"
    elif cfg.get("source", {}).get("cookie"):
        headers["Cookie"] = cfg["source"]["cookie"]

    suffix = ".mp3"
    lowered = url.lower()
    # B 站 m4s 是 AAC 音频流，ffmpeg 能识别，但因无扩展名按 m4a 处理
    for ext in (".flac", ".m4a", ".m4s", ".wav", ".ogg", ".mp3"):
        if ext in lowered:
            suffix = ".m4a" if ext == ".m4s" else ext
            break
    target = dest.with_suffix(suffix)

    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                total = 0
                with target.open("wb") as handle:
                    for chunk in resp.iter_bytes(256 * 1024):
                        total += len(chunk)
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        target.unlink(missing_ok=True)
        raise SourceError(f"音频下载失败: {exc}") from exc

    if not target.exists() or target.stat().st_size < 64 * 1024:
        target.unlink(missing_ok=True)
        raise SourceError("下载的文件过小，可能是无效链接或需要 VIP")
    logger.info("下载完成 %s (%.1f MB)", target.name, target.stat().st_size / 1024 / 1024)
    return target


def resolve(spec: dict, cfg: dict, workdir: Path) -> tuple[Path, dict[str, Any]]:
    """把任务描述解析成音频文件。

    spec 支持三种写法：
      {"audio_path": "/abs/x.mp3"}      本地文件
      {"audio_url": "https://..."}      直链
      {"song": "晴天 周杰伦"}            关键词搜索
    """
    workdir.mkdir(parents=True, exist_ok=True)

    local = spec.get("audio_path")
    if local:
        path = Path(local).expanduser()
        if not path.is_file():
            raise SourceError(f"本地音频不存在: {path}")
        return path, {"title": path.stem, "artist": "", "origin": "local"}

    direct = spec.get("audio_url")
    if direct:
        path = download(direct, workdir / "input", cfg)
        return path, {"title": spec.get("song") or path.stem, "artist": "", "origin": "url"}

    keyword = (spec.get("song") or "").strip()
    if not keyword:
        raise SourceError("缺少 song / audio_url / audio_path")

    limit = int(cfg.get("source", {}).get("search_limit", 5))
    tracks = search(cfg, keyword, limit=limit)

    # 允许指定用第几个候选：某个版本分离出来没人声时，上层可以用下一个候选重试
    idx = int(spec.get("candidate_index") or 0)
    if idx < 0 or idx >= len(tracks):
        idx = 0
    tracks = tracks[idx:] + tracks[:idx]

    # 候选之间依次尝试；每个候选内部再按音源优先级取址
    last_error: Exception | None = None
    for candidate in tracks[: max(1, min(limit, 3))]:
        url, origin, cover = best_url(cfg, candidate)
        if not url:
            last_error = SourceError(f"{candidate.get('title')} 没有可用音源")
            continue
        try:
            path = download(url, workdir / "input", cfg)
            size = path.stat().st_size
            # 小于 2 MB 基本是 60~70 秒的试听片段，换下一个候选；
            # B 站二创常是 1~2 分钟的短片，放宽到 0.5 MB
            min_size = 0.5 * 1024 * 1024 if origin == "bilibili" else 2 * 1024 * 1024
            if size < min_size:
                logger.warning("候选 %s 只有 %.1f MB（疑似试听），换下一个",
                               candidate.get("title"), size / 1048576)
                last_error = SourceError("音频文件过小")
                continue
            info = {
                "title": candidate["title"],
                "artist": candidate["artist"],
                "album": candidate.get("album", ""),
                "origin": origin,
                "size_mb": round(size / 1048576, 1),
                "candidates": len(tracks),
            }
            # 标记"这版可能是翻唱"，插件可据此提示用户
            if cover:
                info["suspect_cover"] = True
                info["cover_note"] = "原唱版本未找到，已使用翻唱音频"
            elif origin == "txqq":
                info["suspect_cover"] = True
            return path, info
        except SourceError as exc:
            last_error = exc
            logger.warning("候选下载失败 %s: %s", candidate["title"], exc)
    raise SourceError(
        f"没找到《{keyword}》的原唱版本（可能只有翻唱）：{last_error}"
    )
