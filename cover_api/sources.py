"""音源解析：多平台并发搜索 -> 精准匹配排序 -> 多源取址与自动 B 站兜底 -> 下载落地。

优化亮点：
1. 支持直接传 B 站 BV 号 / 链接 / b23.tv 短链 / 音频直链。
2. 并发聚合搜索：QQ 官方 + GD 网易云 + txqq 多引擎（酷狗/网易云/QQ）+ 网易云直搜补位，
   速度更快、覆盖率大幅提升（含周杰伦酷狗原唱、独立音乐、二次元/游戏 OST）。
3. 标题相关性硬约束：避免出现搜「speed of light 塞壬唱片」却误命中同歌手另一首歌的问题。
4. 仅剥离平台路由词（如「B站」「bilibili」），保留「司凤」「鬼畜」等真实内容词。
5. 智能试听片段识别：小文件用 ffprobe 校验时长，拒绝 30s VIP 试听而不误杀 2 分钟完整短歌。
6. 全能 B 站自动兜底：当音乐平台无结果或全为 VIP 试听时，自动以高质音乐评分策略从 B 站提取
   DASH 纯音频流（支持 backupUrl 多 CDN 镜像），尽量让每一首歌都能搜出完整原唱。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any
import urllib.parse

import httpx

from . import qq_recording

logger = logging.getLogger("cover.source")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://music.txqq.pro",
    "Referer": "https://music.txqq.pro",
}
GD_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
}
BILI_HEADERS = {
    "User-Agent": UA,
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
_BILI_BUVID = "B1A2C3D4-E5F6-7890-ABCD-EF1234567890infoc"

# 平台路由词（触发 B 站优先，且在发往 B 站搜索前应剥离）
_BILI_PLATFORM_WORDS = ("b站", "bilibili", "B站", "哔哩哔哩")

# 内容特征词（触发 B 站优先，但必须保留在搜索词中，绝不能剥离！）
_BILI_CONTENT_HINTS = (
    "司凤", "鬼畜", "空耳", "二创", "邦吧", "dj版", "phonk",
    "米塔", "公开发疯", "手书", "填词", "拼好歌", "ai翻唱",
)

# 兼容旧引用
_BILI_HINTS = _BILI_PLATFORM_WORDS + _BILI_CONTENT_HINTS

_BAD_WORDS = (
    "伴奏", "翻唱", "纯音乐", "remix", "dj版", "dj舞曲", "dj加快", "instrumental", "karaoke",
    "片段", "铃声", "童声", "儿歌", "钢琴版", "钢琴曲", "吉他版", "吉他弹唱", "古筝版", "二胡版", "竹笛", "笛子版",
    "八音盒", "交响", "慢摇", "喊麦", "demo", "live", "现场", "演唱会",
    "女声版", "男声版", "合唱", "串烧", "抖音", "翻自", "cover", "纯享",
    "鬼畜", "拼好歌", "纯伴奏", "off vocal", "inst版", "(inst", "（inst",
    "inst.", "minus one", "伴唱", "消音",
)

# B 站通用兜底额外过滤词（非歌曲类视频）
_BILI_NOISE_WORDS = (
    "教程", "教学", "鼓谱", "谱子", "吉他谱", "钢琴谱", "简谱", "五线谱",
    "reaction", "反应", "解析", "乐评", "鉴赏", "合集", "歌单", "循环",
    "1小时", "一小时", "半小时", "混剪", "踩点", "卡点", "剪辑", "鬼畜调教",
    "清唱教学", "翻弹", "指弹", "架子鼓", "贝斯cover", "吉他cover", "钢琴版",
)

# B 站高质原曲加分信号
_BILI_GOOD_WORDS = (
    "hi-res", "无损", "原唱", "官方", "mv", "完整版", "纯享", "歌词",
    "音质", "flac", "音源", "ep", "official",
)

_ORIGIN_LABELS = {
    "qq": "QQ音乐",
    "netease": "网易云",
    "kugou": "酷狗",
    "kuwo": "酷我",
    "txqq": "聚合源",
    "bilibili": "B站",
    "meting": "网易云",
    "local": "本地",
    "url": "直链",
}

_BV_RE = re.compile(r"\b(BV[0-9A-Za-z]{10})\b")
_B23_RE = re.compile(r"https?://b23\.tv/[0-9A-Za-z]+", re.I)
_DIRECT_AUDIO_RE = re.compile(r"^https?://\S+\.(?:mp3|wav|m4a|flac|ogg)(?:\?\S*)?$", re.I)


class SourceError(RuntimeError):
    """音源不可用或找不到歌曲。"""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _clean_keyword(text: str) -> str:
    """去掉书名号/引号等装饰符，把「歌手 - 歌名」规范成空格分隔。"""
    s = (text or "").strip()
    s = re.sub(r"[《》「」『』【】\"'“”‘’]", " ", s)
    s = re.sub(r"\s+[-—–/|]\s+", " ", s)
    return _clean(s)


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = (text.replace("&quot;", '"').replace("&amp;", "&")
            .replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'"))
    return _clean(text)


def _extract_bvid(text: str) -> str:
    """从关键词或 URL 中提取 BV 号；支持 b23.tv 短链自动展开。"""
    s = (text or "").strip()
    m = _BV_RE.search(s)
    if m:
        return m.group(1)
    m2 = _B23_RE.search(s)
    if m2:
        try:
            with httpx.Client(timeout=8.0, headers={"User-Agent": UA}, follow_redirects=True) as client:
                resp = client.head(m2.group(0))
                m3 = _BV_RE.search(str(resp.url))
                if m3:
                    return m3.group(1)
        except httpx.HTTPError:
            pass
    return ""


def _wants_bilibili(keyword: str) -> bool:
    low = (keyword or "").lower()
    if _BV_RE.search(keyword or "") or _B23_RE.search(keyword or ""):
        return True
    return any(h.lower() in low for h in _BILI_HINTS)


def _query_variants(keyword: str) -> list[str]:
    """构造合理的搜索变体：保留整句、去括号后缀、前段歌名；绝不单独把歌手名当歌名搜。"""
    cleaned = _clean_keyword(keyword)
    variants: list[str] = [cleaned]
    # 去掉括号里的版本/备注（如「大鱼 (唱片版)」→「大鱼」）
    no_paren = _clean(re.sub(r"[\(（\[【][^\)）\]】]*[\)）\]】]", " ", cleaned))
    if no_paren and no_paren not in variants:
        variants.append(no_paren)
    parts = [p for p in cleaned.replace("　", " ").split(" ") if p]
    if len(parts) == 2:
        # 两个词时（常见「歌名 歌手」或「歌手 歌名」），只补前词作为歌名尝试，不单独搜末尾词避免误命中同歌手其他歌
        if parts[0] not in variants:
            variants.append(parts[0])
    elif len(parts) > 2:
        # 多词（常见英文歌名 + 中文歌手，如「speed of light 塞壬唱片」）：去掉最后一个词作为纯歌名
        head = " ".join(parts[:-1])
        if head not in variants:
            variants.append(head)
    return variants


def _version_tokens(keyword: str) -> list[str]:
    """抽出用户显式指定的版本标记，例如「司凤版」「女声版」「Live 版」。"""
    return [m.group(0) for m in re.finditer(r"[\u4e00-\u9fffA-Za-z0-9]{1,8}版", keyword or "")]


def _exempt_words(keyword: str) -> set[str]:
    """关键词里自己提到的"坏词"要豁免，否则用户点名要现场版反而被过滤掉。"""
    low = (keyword or "").lower()
    asked = {w for w in _BAD_WORDS if w in low}
    asked |= {w for w in _BILI_NOISE_WORDS if w in low}
    if _version_tokens(keyword):
        asked |= {"翻唱", "cover", "女声版", "男声版", "live", "现场", "纯享"}
    return asked


def _norm_name(text: str) -> str:
    """去掉空格、括号内常见噪音标注与标点，便于跨平台比对。"""
    s = (text or "").lower()
    s = re.sub(r"[\s,，、/&·\-—–_:：!！?？.。'\"`~()（）\[\]【】《》]+", "", s)
    return s


def _strip_title_suffix(text: str) -> str:
    """去掉歌名后的括号版本后缀（用于严格比对主歌名）。"""
    return _clean(re.sub(r"[\(（\[【][^\)）\]】]*[\)）\]】]", " ", text or ""))


def _char_sim(a: str, b: str) -> float:
    """字符重合度（Jaccard），跨繁简差异（周杰倫 vs 周杰伦）。"""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa), len(sb))


def _name_match(expected: str, got: str, threshold: float = 0.6) -> bool:
    exp, g = _norm_name(expected), _norm_name(got)
    if not exp or not g:
        return False
    if exp == g or exp in g or g in exp:
        return True
    return _char_sim(exp, g) >= threshold


def _artist_match(expected: str, candidates: Any) -> bool:
    exp_list = [x.strip() for x in re.split(r"[,，、/&]+", expected or "") if x.strip()]
    if not exp_list:
        return True
    if isinstance(candidates, str):
        candidates = [x.strip() for x in re.split(r"[,，、/&]+", candidates) if x.strip()]
    cand_list = [str(x) for x in (candidates or []) if x]
    if not cand_list:
        return False
    return any(_name_match(e, c) for e in exp_list for c in cand_list)


def _album_match(expected: str, got: str) -> bool:
    exp = _norm_name(_strip_title_suffix(expected))
    if not exp:
        return True
    g = _norm_name(_strip_title_suffix(got))
    if not g:
        return False
    return exp == g or exp in g or g in exp or _char_sim(exp, g) >= 0.55


def _title_relevance(title: str, artist: str, keyword: str, tokens: list[str]) -> float:
    """估算候选歌名与查询的匹配度（0.0 ~ 1.0）。

    关键修正：把已经命中 artist 的 token 排除后，剩下的 token 必须至少部分命中 title，
    否则就会把同歌手的另一首歌（如搜「speed of light 塞壬唱片」命中「A Necessary Message」）误判高分。
    """
    norm_t = _norm_name(title)
    norm_t_core = _norm_name(_strip_title_suffix(title))
    norm_a = _norm_name(artist)
    norm_k = _norm_name(keyword)
    if not norm_t or not norm_k:
        return 0.0
    if norm_t == norm_k or norm_t_core == norm_k:
        return 1.0

    # 区分「可能是歌手名的词」和「应该是歌名的词」
    song_tokens: list[str] = []
    for tok in tokens:
        nt = _norm_name(tok)
        if not nt:
            continue
        if nt in _BILI_PLATFORM_WORDS:
            continue
        # 若该词完全命中歌手且未命中歌名，它大概率是歌手名
        if norm_a and (nt in norm_a or _char_sim(nt, norm_a) >= 0.6) and nt not in norm_t:
            continue
        song_tokens.append(nt)

    if not song_tokens:
        # 全部 token 都像歌手名（用户可能只输入了单字/词，同时在歌手里）
        song_tokens = [_norm_name(t) for t in tokens if _norm_name(t)]

    # 拼成歌名串做整体包含与字符覆盖度检查
    joined = "".join(song_tokens)
    if joined and (joined in norm_t or norm_t_core in joined):
        return 0.95

    hit_tokens = 0
    for st in song_tokens:
        if st in norm_t or (len(st) >= 2 and _char_sim(st, norm_t_core) >= 0.55):
            hit_tokens += 1
    if song_tokens and hit_tokens > 0:
        return min(0.9, 0.35 + 0.55 * (hit_tokens / len(song_tokens)))

    # 兜底：字符级覆盖（兼容繁简差异，如 铁花飞 vs 鐵花飛）
    if joined:
        common = len(set(joined) & set(norm_t_core))
        ratio = common / max(1, len(set(joined)))
        if ratio >= 0.5:
            return round(0.45 * ratio + 0.2, 2)
    return 0.0


def _has_bad_word(track: dict[str, Any], exempt: set[str] | None = None) -> bool:
    text = (track.get("title", "") + " " + track.get("artist", "")).lower()
    exempt = exempt or set()
    return any(word in text for word in _BAD_WORDS if word not in exempt)


def _score(track: dict[str, Any], keyword: str, tokens: list[str],
           exempt: set[str] | None = None) -> int:
    """给候选打分：强约束歌名相关性 + 歌手加分 + 版本词加分 + 排掉伴奏/翻唱。"""
    title = (track.get("title") or "").lower()
    artist = (track.get("artist") or "").lower()
    key = _clean_keyword(keyword).lower()
    exempt = exempt or set()
    score = 0

    rel = _title_relevance(title, artist, key, tokens)
    if rel >= 0.95:
        score += 120
    elif rel >= 0.6:
        score += int(rel * 100)
    elif rel >= 0.25:
        score += int(rel * 60)
    else:
        # 歌名几乎零重合：重罚，防止同歌手无关歌曲抢占高位
        score -= 160

    if _norm_name(title) == _norm_name(key):
        score += 40
    elif key in title:
        score += 20
    # 干净主歌名（无括号后缀如 "(旧版)"）加分，优先排在括号变体前面
    if "(" not in title and "（" not in title and "[" not in title and "【" not in title:
        score += 14

    for token in tokens:
        low = token.lower()
        if low and low in artist:
            score += 65 if rel >= 0.25 else 10
        if low and low in title:
            score += 12

    for ver in _version_tokens(keyword):
        if ver.lower() in title:
            score += 120

    for bad in _BAD_WORDS:
        if bad in exempt:
            continue
        if bad in title or bad in artist:
            score -= 55

    # 来源可靠度微调（同分时优先有直链/官方元数据/时长正常的）
    dur = int(track.get("duration") or 0)
    if 60 <= dur <= 420:
        score += 8
    elif 0 < dur < 45:
        score -= 80
    if track.get("url") or track.get("gd_id") or track.get("bvid") or track.get("songmid"):
        score += 10
    if track.get("origin") == "bilibili":
        for nw in _BILI_NOISE_WORDS:
            if nw not in exempt and nw in title:
                score -= 70
        for gw in _BILI_GOOD_WORDS:
            if gw in title:
                score += 15
    return score


# ---------------------------------------------------------------- B 站音源
def _bili_client(cfg: dict) -> httpx.Client:
    source = cfg.get("source", {})
    client = httpx.Client(timeout=18.0, headers=dict(BILI_HEADERS), follow_redirects=True)
    user_cookie = (source.get("bili_cookie") or "").strip()
    if user_cookie:
        client.headers["Cookie"] = user_cookie
    return client


def _bili_duration(value: Any) -> int:
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


def _bili_track_from_bvid(cfg: dict, bvid: str) -> dict[str, Any] | None:
    """直接按 BV 号查视频元信息。"""
    view_api = cfg.get("source", {}).get("bili_view_api") or "https://api.bilibili.com/x/web-interface/view"
    referer = f"https://www.bilibili.com/video/{bvid}"
    try:
        with _bili_client(cfg) as client:
            with contextlib.suppress(httpx.HTTPError):
                client.get(referer, timeout=8.0)
            resp = client.get(view_api, params={"bvid": bvid}, headers={"Referer": referer})
            resp.raise_for_status()
            payload = resp.json()
        if payload.get("code") != 0:
            return None
        d = payload.get("data") or {}
        return {
            "title": _strip_html(d.get("title") or bvid),
            "artist": _clean((d.get("owner") or {}).get("name") or ""),
            "bvid": bvid,
            "duration": int(d.get("duration") or 0),
            "url": "",
            "id": bvid,
            "origin": "bilibili",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("B 站查询 %s 失败: %s", bvid, exc)
        return None


def _search_bilibili(cfg: dict, keyword: str, limit: int = 5, *, general_music: bool = False) -> list[dict[str, Any]]:
    """B 站视频搜索。

    关键修正：只剥离平台路由词（「B站」「bilibili」），绝不剥离「司凤」「鬼畜」等内容词！
    general_music=True 时用于通用兜底搜歌，自动过滤非音乐类噪音与过长/过短视频。
    """
    bvid_direct = _extract_bvid(keyword)
    if bvid_direct:
        track = _bili_track_from_bvid(cfg, bvid_direct)
        if track:
            return [track]

    source = cfg.get("source", {})
    api = source.get("bili_search_api") or "https://api.bilibili.com/x/web-interface/search/type"
    max_sec = int((cfg.get("limits") or {}).get("max_song_seconds", 420))

    q = _clean_keyword(keyword)
    for pw in _BILI_PLATFORM_WORDS:
        q = re.sub(re.escape(pw), " ", q, flags=re.I)
    q = _clean(q) or _clean_keyword(keyword)

    with _bili_client(cfg) as client:
        with contextlib.suppress(httpx.HTTPError):
            client.get("https://www.bilibili.com/", timeout=8.0)
        resp = client.get(
            api,
            params={"search_type": "video", "keyword": q, "page": 1},
            headers={"Referer": "https://www.bilibili.com/"},
        )
        resp.raise_for_status()
        payload = resp.json()
    if payload.get("code") != 0:
        logger.warning("B 站搜索失败：%s", payload.get("message"))
        return []

    results = ((payload.get("data") or {}).get("result") or [])
    exempt = _exempt_words(q)
    tokens = [t for t in q.split(" ") if t]

    candidates: list[dict[str, Any]] = []
    for item in results:
        bvid = (item.get("bvid") or "").strip()
        if not bvid:
            continue
        dur = _bili_duration(item.get("duration"))
        # 过滤过短（<40s 片段）或超长（如 1 小时循环歌单）
        if dur and (dur < 40 or dur > max_sec + 60):
            continue
        title = _strip_html(item.get("title") or "")
        artist = _clean(item.get("author") or "")
        candidates.append({
            "title": title,
            "artist": artist,
            "bvid": bvid,
            "duration": dur,
            "url": "",
            "id": bvid,
            "origin": "bilibili",
        })

    if not candidates:
        return []

    # 综合打分：既照顾二创版本词命中，也过滤钢琴教程/鼓谱/循环歌单
    def _bili_rank(t: dict[str, Any]) -> tuple[int, int]:
        tl = t["title"].lower()
        noise = any(w in tl for w in _BILI_NOISE_WORDS if w not in exempt)
        bad = _has_bad_word(t, exempt) if general_music else False
        sc = _score(t, q, tokens, exempt)
        # 额外奖励命中用户显式内容词（司凤/鬼畜/空耳等）
        for h in _BILI_CONTENT_HINTS:
            if h in q.lower() and h in tl:
                sc += 90
        return (1 if (noise or bad) else 0, -sc)

    candidates.sort(key=_bili_rank)
    return candidates[:limit]


def _bili_audio_urls(cfg: dict, bvid: str) -> list[str]:
    """bvid -> cid -> 返回 DASH 纯音频流直链列表（含 backupUrl 镜像）。"""
    source = cfg.get("source", {})
    view_api = source.get("bili_view_api") or "https://api.bilibili.com/x/web-interface/view"
    play_api = source.get("bili_play_api") or "https://api.bilibili.com/x/player/playurl"
    referer = f"https://www.bilibili.com/video/{bvid}"

    for attempt in range(3):
        try:
            with _bili_client(cfg) as client:
                with contextlib.suppress(httpx.HTTPError):
                    client.get(referer, timeout=8.0)

                r1 = client.get(view_api, params={"bvid": bvid}, headers={"Referer": referer})
                r1.raise_for_status()
                view = r1.json()
                if view.get("code") != 0:
                    return []
                cid = (view.get("data") or {}).get("cid")
                if not cid:
                    return []

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
            audios = (data.get("dash") or {}).get("audio") or []
            if audios:
                best = max(audios, key=lambda a: int(a.get("bandwidth") or 0))
                urls: list[str] = []
                primary = best.get("baseUrl") or best.get("base_url") or ""
                if primary:
                    urls.append(primary)
                for bu in (best.get("backupUrl") or best.get("backup_url") or []):
                    if bu and bu not in urls:
                        urls.append(bu)
                if urls:
                    return urls
            durl = data.get("durl") or []
            if durl and durl[0].get("url"):
                return [durl[0]["url"]]
            return []
        except httpx.HTTPError as exc:
            logger.warning("B 站取址失败（第 %d 次）：%s", attempt + 1, exc)
            continue
    return []


def _bili_audio_url(cfg: dict, bvid: str) -> str:
    """兼容旧接口：返回首条 B 站音频流直链。"""
    urls = _bili_audio_urls(cfg, bvid)
    return urls[0] if urls else ""


# ---------------------------------------------------------------- QQ 官方接口
def _qq_cookie(cfg: dict) -> str:
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
    return qq_recording.account_id(cookie)


def _search_qq(cfg: dict, keyword: str, limit: int = 6) -> list[dict[str, Any]]:
    cookie = _qq_cookie(cfg)
    headers = {"User-Agent": UA, "Referer": "https://y.qq.com/"}
    if cookie:
        headers["Cookie"] = cookie

    resp = httpx.get(
        "https://c.y.qq.com/soso/fcgi-bin/client_search_cp",
        params={"w": keyword, "format": "json", "n": max(limit, 10), "p": 1,
                "cr": 1, "g_tk": 5381, "t": 0},
        headers=headers, timeout=12.0, follow_redirects=True,
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
            "id": f"qq:{mid}",
            "origin": "qq",
        })
    return out


def _qq_audio_url(cfg: dict, songmid: str) -> str:
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
                                   "User-Agent": UA}, timeout=12.0)
    resp.raise_for_status()
    info = ((resp.json().get("req_0") or {}).get("data") or {}).get("midurlinfo") or [{}]
    purl = (info[0].get("purl") or "").strip()
    return ("https://ws.stream.qqmusic.qq.com/" + purl) if purl else ""


# ---------------------------------------------------------------- GD 聚合源（网易云）
def _gd_settings(cfg: dict) -> tuple[str, list[str]]:
    source_cfg = cfg.get("source", {}) or {}
    api = source_cfg.get("gd_api") or "https://music-api.gdstudio.xyz/api.php"
    # 默认仅保留实测可用的 netease（原 kuwo 已报 unsupported、joox 全返空 URL）
    raw_subs = source_cfg.get("gd_sources") or ["netease"]
    subs = [s for s in raw_subs if s not in ("kuwo", "joox")] or ["netease"]
    return api, subs


def _gd_query(title: str, artist: str) -> str:
    return f"{title} {artist.split(',')[0]}".strip() if artist else title


def _gd_url_by_id(cfg: dict, source: str, song_id: Any) -> str:
    """按音质降级尝试取直链（320 -> 192 -> 128）。"""
    api, _ = _gd_settings(cfg)
    for br in (320, 192, 128):
        try:
            r = httpx.get(api, params={"types": "url", "source": source, "id": song_id, "br": br},
                          headers=GD_HEADERS, timeout=12.0)
            data = r.json()
            url = (data.get("url") or "") if isinstance(data, dict) else ""
            if url:
                return url
        except (httpx.HTTPError, ValueError):
            continue
    return ""


def _search_gd(cfg: dict, keyword: str, limit: int = 6) -> list[dict[str, Any]]:
    """直接用 GD Studio 搜网易云曲库，补足 QQ 搜不到或独家的歌曲（二次元/独立音乐/塞壬唱片等）。"""
    api, subs = _gd_settings(cfg)
    out: list[dict[str, Any]] = []
    for sub in subs:
        try:
            r = httpx.get(api, params={"types": "search", "source": sub, "name": keyword,
                                       "count": max(limit, 8), "pages": 1},
                          headers=GD_HEADERS, timeout=12.0)
            hits = r.json()
        except (httpx.HTTPError, ValueError):
            continue
        if not isinstance(hits, list):
            continue
        for h in hits:
            sid = h.get("id")
            name = _clean(h.get("name") or "")
            if not sid or not name:
                continue
            art = h.get("artist")
            artist = _clean(",".join(str(a) for a in art) if isinstance(art, list) else str(art or ""))
            out.append({
                "title": name,
                "artist": artist,
                "album": _clean(str(h.get("album") or "")),
                "duration": 0,
                "gd_id": str(sid),
                "gd_source": sub,
                "netease_id": str(sid) if sub == "netease" else "",
                "url": "",
                "id": f"gd:{sub}:{sid}",
                "origin": sub,
            })
    return out


def _gd_pick(cfg: dict, source: str, title: str, artist: str,
             album: str = "") -> tuple[str, str]:
    """在指定 GD 音源里按两级精度匹配原唱：
    1) 歌手 + 专辑双匹配
    2) 歌手 + 主歌名精准匹配（且不含翻唱/伴奏坏词），兼容跨平台专辑名差异
    """
    api, _ = _gd_settings(cfg)
    name = _gd_query(title, artist)
    try:
        r = httpx.get(api, params={"types": "search", "source": source,
                                   "name": name, "count": 10, "pages": 1},
                      headers=GD_HEADERS, timeout=12.0)
        hits = r.json()
    except (httpx.HTTPError, ValueError):
        return "", ""
    if not isinstance(hits, list) or not hits:
        return "", ""

    exempt = _exempt_words(title)
    tier1 = [
        h for h in hits
        if _artist_match(artist, h.get("artist"))
        and _album_match(album, str(h.get("album") or ""))
        and not _has_bad_word({"title": str(h.get("name") or ""), "artist": ""}, exempt)
    ]
    tier2 = [
        h for h in hits
        if _artist_match(artist, h.get("artist"))
        and _name_match(_strip_title_suffix(title), _strip_title_suffix(str(h.get("name") or "")), 0.7)
        and not _has_bad_word({"title": str(h.get("name") or ""), "artist": ""}, exempt)
    ]
    matched = tier1 or tier2
    if not matched:
        logger.info("音源 %s 没有《%s》的原唱版本（歌手/歌名对不上）", source, title)
        return "", ""

    for hit in matched[:3]:
        sid = hit.get("id")
        if not sid:
            continue
        url = _gd_url_by_id(cfg, source, sid)
        if url:
            return url, source
    return "", ""


def _gd_lenient(cfg: dict, title: str, artist: str) -> tuple[str, str]:
    """严格校验失败后的兜底：优先不含伴奏/纯音乐且歌名相关的首条结果。"""
    api, subs = _gd_settings(cfg)
    q = _gd_query(title, artist)
    exempt = _exempt_words(title)
    for source in subs:
        try:
            r = httpx.get(api, params={"types": "search", "source": source,
                                       "name": q, "count": 8, "pages": 1},
                          headers=GD_HEADERS, timeout=12.0)
            hits = r.json()
            if not isinstance(hits, list) or not hits:
                continue
            # 过滤掉纯伴奏/歌名完全无关的条目
            good = [
                h for h in hits
                if _name_match(_strip_title_suffix(title), _strip_title_suffix(str(h.get("name") or "")), 0.5)
                and not any(bw in str(h.get("name") or "").lower()
                            for bw in ("伴奏", "instrumental", "纯音乐", "off vocal", "inst.") if bw not in exempt)
            ]
            for hit in (good or hits)[:3]:
                sid = hit.get("id")
                if not sid:
                    continue
                url = _gd_url_by_id(cfg, source, sid)
                if url:
                    logger.info("音源 %s 使用候选兜底：%s", source, hit.get("name"))
                    return url, source
        except (httpx.HTTPError, ValueError):
            continue
    return "", ""


def _gd_audio_url(cfg: dict, title: str, artist: str, album: str = "") -> tuple[str, str, bool]:
    _, subs = _gd_settings(cfg)
    for source in subs:
        url, _ = _gd_pick(cfg, source, title, artist, album)
        if url:
            logger.info("音源 %s 命中《%s》原唱", source, title)
            return url, source, False

    if not cfg.get("source", {}).get("allow_cover_fallback", True):
        return "", "", False

    url, source = _gd_lenient(cfg, title, artist)
    return url, source, bool(url)


# ---------------------------------------------------------------- txqq 多引擎聚合
def _search_txqq(cfg: dict, keyword: str, limit: int = 6) -> list[dict[str, Any]]:
    """并发搜索 txqq 多平台子源（默认 kugou + netease），快速拿到直链候选。"""
    source = cfg.get("source", {})
    base = source.get("txqq_base") or "https://music.txqq.pro/"
    txqq_types = [t for t in (source.get("txqq_types") or ["kugou", "netease"]) if t != "qq"] or ["kugou", "netease"]
    headers = dict(HEADERS)
    if source.get("cookie"):
        headers["Cookie"] = source["cookie"]

    cleaned = _clean_keyword(keyword)
    tokens = [t for t in cleaned.split(" ") if t]
    exempt = _exempt_words(cleaned)
    collected: dict[str, dict[str, Any]] = {}

    def _query_one(tp: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            resp = httpx.post(
                base,
                data={"input": cleaned, "filter": "name", "type": tp, "page": 1},
                headers=headers,
                timeout=6.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError):
            return rows
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return rows
        for item in data:
            url = (item.get("url") or item.get("link") or "").strip()
            title = _clean(item.get("title") or "")
            if not url or not title:
                continue
            sid = str(item.get("songid") or url[:32])
            rows.append({
                "title": title,
                "artist": _clean(item.get("author") or ""),
                "album": "",
                "duration": 0,
                "url": url,
                "cover": item.get("pic") or "",
                "id": f"txqq:{tp}:{sid}",
                "origin": tp if tp in ("kugou", "netease", "qq") else "txqq",
                "txqq_type": tp,
                "netease_id": sid if tp == "netease" else "",
            })
        return rows

    with ThreadPoolExecutor(max_workers=len(txqq_types)) as pool:
        for res in pool.map(_query_one, txqq_types):
            for row in res:
                collected.setdefault(row["id"], row)

    if not collected:
        return []

    ranked = sorted(
        collected.values(),
        key=lambda t: (_has_bad_word(t, exempt), -_score(t, cleaned, tokens, exempt)),
    )
    kept = [t for t in ranked if not _has_bad_word(t, exempt)]
    return (kept or ranked)[:limit]


# ---------------------------------------------------------------- 网易云官方直搜 + Meting 兜底
def _search_netease_direct(keyword: str, limit: int = 5) -> list[dict[str, Any]]:
    """网易云公开云搜索（仅元数据，可搭配 GD/Meting 取直链）。"""
    try:
        r = httpx.post(
            "https://music.163.com/api/cloudsearch/pc",
            data={"s": keyword, "type": 1, "limit": max(limit, 6), "offset": 0},
            headers={"User-Agent": UA, "Referer": "https://music.163.com/"},
            timeout=10.0,
        )
        r.raise_for_status()
        songs = (r.json().get("result") or {}).get("songs") or []
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    for s in songs:
        sid = s.get("id")
        name = _clean(s.get("name") or "")
        if not sid or not name:
            continue
        ars = [str(a.get("name") or "") for a in (s.get("ar") or []) if isinstance(a, dict)]
        out.append({
            "title": name,
            "artist": _clean(",".join(a for a in ars if a)),
            "album": _clean((s.get("al") or {}).get("name") or ""),
            "duration": int((s.get("dt") or 0) // 1000),
            "gd_id": str(sid),
            "gd_source": "netease",
            "netease_id": str(sid),
            "url": "",
            "id": f"netease:{sid}",
            "origin": "netease",
        })
    return out[:limit]


def _meting_audio_url(netease_id: str) -> str:
    if not netease_id:
        return ""
    return f"https://api.i-meto.com/meting/api?server=netease&type=url&id={urllib.parse.quote(str(netease_id))}"


# ---------------------------------------------------------------- 取址策略
def best_url(cfg: dict, track: dict[str, Any]) -> tuple[str, str, bool]:
    """为单个候选按优先级解析音频直链，返回 (url, 来源标记, 是否翻唱兜底)。"""
    source = cfg.get("source", {})
    order = source.get("order") or ["bilibili", "gd", "txqq", "qq"]
    mid = track.get("songmid") or ""

    # 1. B 站候选必须直接走 B 站取址
    if track.get("bvid"):
        u = _bili_audio_url(cfg, track["bvid"])
        if u:
            return u, "bilibili", False

    # 2. 候选自带已匹配的 GD 网易云 ID，直接取直链（免去二次搜索误匹配）
    if track.get("gd_id"):
        sub = track.get("gd_source") or "netease"
        u = _gd_url_by_id(cfg, sub, track["gd_id"])
        if u:
            return u, sub, False

    # 3. 候选自带可用直链（如 txqq kugou/netease），若无更高优源直接可用
    direct_url = (track.get("url") or "").strip()
    if direct_url and track.get("origin") in ("kugou", "netease", "txqq"):
        return direct_url, track.get("origin") or "txqq", False

    for kind in order:
        try:
            if kind == "qq" and mid:
                u = _qq_audio_url(cfg, mid)
                if u:
                    return u, "qq", False
            elif kind == "gd":
                u, src, cover = _gd_audio_url(
                    cfg, track.get("title", ""), track.get("artist", ""), track.get("album", ""),
                )
                if u:
                    return u, src or "netease", cover
            elif kind == "txqq" and direct_url:
                return direct_url, track.get("origin") or "txqq", False
        except Exception as exc:  # noqa: BLE001
            logger.warning("音源 %s 取址失败：%s", kind, exc)

    # 4. 仍有 netease_id 时尝试 Meting 兜底
    if track.get("netease_id") and source.get("meting_enabled", True):
        return _meting_audio_url(track["netease_id"]), "meting", False

    return "", "", False


# ---------------------------------------------------------------- 并发综合搜索
def search(cfg: dict, keyword: str, limit: int = 8, *, source_hint: str = "auto") -> list[dict[str, Any]]:
    """多平台并发搜索 + 去重 + 强相关性排序。

    source_hint:
      - "auto": 自动判断（命中 B 站特征词或 BV 号走 B 站，否则并发聚合全音乐平台）
      - "bilibili": 强制搜 B 站
      - "music": 仅搜音乐平台（QQ + GD 网易云 + txqq 酷狗/网易云/QQ）
    """
    raw_kw = _clean(keyword)
    if not raw_kw:
        raise SourceError("关键词为空")

    source_cfg = cfg.get("source", {})
    bili_enabled = bool(source_cfg.get("bili_enabled", True))

    # 1. 直接 BV 号 / b23.tv 或显式要求 B 站
    if source_hint == "bilibili" or (
        source_hint == "auto" and _wants_bilibili(raw_kw) and bili_enabled
    ):
        try:
            bili = _search_bilibili(cfg, raw_kw, limit, general_music=(source_hint == "bilibili" and not _wants_bilibili(raw_kw)))
            if bili:
                logger.info("B 站命中《%s》%d 条视频", raw_kw, len(bili))
                return bili
        except Exception as exc:  # noqa: BLE001
            logger.warning("B 站搜索失败，回退音乐平台：%s", exc)
        if source_hint == "bilibili":
            raise SourceError(f"B 站没搜到《{raw_kw}》")

    cleaned = _clean_keyword(raw_kw)
    tokens = [t for t in cleaned.split(" ") if t]
    exempt = _exempt_words(cleaned)

    # 2. 并发查询三大音乐源：QQ 官方、GD 网易云、txqq（酷狗/网易云/QQ）
    tasks = {
        "qq": lambda: _search_qq(cfg, cleaned, max(limit, 8)),
        "gd": lambda: _search_gd(cfg, cleaned, max(limit, 8)),
        "txqq": lambda: _search_txqq(cfg, cleaned, max(limit, 8)),
    }
    tracks: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                res = fut.result()
                if res:
                    tracks.extend(res)
            except Exception as exc:  # noqa: BLE001
                logger.debug("音源 %s 搜索异常: %s", name, exc)

    # 若三大源都空，再补一次网易云官方云搜索 + B 站通用搜索
    if not tracks:
        tracks.extend(_search_netease_direct(cleaned, limit))
    if not tracks and bili_enabled and source_hint != "music":
        with contextlib.suppress(Exception):
            tracks.extend(_search_bilibili(cfg, cleaned, limit, general_music=True))

    if not tracks:
        raise SourceError(f"全平台都没搜到《{raw_kw}》，可以换个写法或直接贴 B 站 BV 号")

    # 3. 合并去重：同名同歌手保留信息最全的一条（同时把 gd_id/songmid/url 合并进去！）
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for t in tracks:
        key = (_norm_name(t.get("title") or ""), _norm_name((t.get("artist") or "").split(",")[0]))
        if key not in merged:
            merged[key] = dict(t)
        else:
            existing = merged[key]
            for field in ("songmid", "gd_id", "gd_source", "netease_id", "url", "album", "bvid"):
                if not existing.get(field) and t.get(field):
                    existing[field] = t[field]
            if not existing.get("duration") and t.get("duration"):
                existing["duration"] = t["duration"]

    unique = list(merged.values())
    ranked = sorted(
        unique,
        key=lambda t: (
            _has_bad_word(t, exempt),
            0 if _title_relevance(t.get("title", ""), t.get("artist", ""), cleaned, tokens) >= 0.25 else 1,
            -_score(t, cleaned, tokens, exempt),
        ),
    )

    # 优先保留「非坏词且歌名相关」的候选
    good = [
        t for t in ranked
        if not _has_bad_word(t, exempt)
        and _title_relevance(t.get("title", ""), t.get("artist", ""), cleaned, tokens) >= 0.25
    ]

    # 如果音乐平台压根没有歌名相关的结果，自动混入 B 站通用搜索结果救场！
    if not good and bili_enabled and source_hint != "music":
        try:
            bili_rescue = _search_bilibili(cfg, cleaned, min(limit, 5), general_music=True)
            if bili_rescue:
                logger.info("音乐平台未命中高相关歌曲，B 站救场命中 %d 条", len(bili_rescue))
                return bili_rescue[:limit]
        except Exception as exc:  # noqa: BLE001
            logger.debug("B 站救场搜索失败: %s", exc)

    kept = good or [t for t in ranked if not _has_bad_word(t, exempt)] or ranked
    return kept[:limit]


def search_preview(cfg: dict, keyword: str, limit: int = 8, *, source_hint: str = "auto") -> list[dict[str, Any]]:
    """供 WebUI 试听工坊展示的候选列表（带中文音源标签与时长）。"""
    tracks = search(cfg, keyword, limit=limit, source_hint=source_hint)
    out: list[dict[str, Any]] = []
    for i, t in enumerate(tracks):
        origin = str(t.get("origin") or "txqq")
        out.append({
            "index": i,
            "id": str(t.get("id") or i),
            "title": t.get("title") or "",
            "artist": t.get("artist") or "",
            "album": t.get("album") or "",
            "duration": int(t.get("duration") or 0),
            "origin": origin,
            "origin_label": _ORIGIN_LABELS.get(origin, origin),
            "bvid": t.get("bvid") or "",
            "gd_id": t.get("gd_id") or "",
            "gd_source": t.get("gd_source") or "",
            "songmid": t.get("songmid") or "",
            "netease_id": t.get("netease_id") or "",
            "url": t.get("url") or "",
        })
    return out


# ---------------------------------------------------------------- 下载与智能试听判定
def _probe_audio_duration(path: Path, cfg: dict) -> float:
    ffprobe = (cfg.get("mix") or {}).get("ffprobe", "ffprobe")
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        return float((proc.stdout or "0").strip() or 0)
    except Exception:  # noqa: BLE001
        return 0.0


def download(url: str, dest: Path, cfg: dict, *, qq_media: bool = False) -> Path:
    """下载音频到 dest（自动补扩展名，带短暂重试）。返回实际路径。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": UA, "Referer": "https://music.txqq.pro/"}
    if qq_media:
        headers["Referer"] = "https://y.qq.com/"
    elif "bilibili" in url or "bilivideo" in url or "akamaized" in url:
        headers["Referer"] = "https://www.bilibili.com/"
        headers["Origin"] = "https://www.bilibili.com"
        headers["Cookie"] = cfg.get("source", {}).get("bili_cookie") or f"buvid3={_BILI_BUVID};"
    elif "qqmusic" in url or "y.qq.com" in url:
        headers["Referer"] = "https://y.qq.com/"
    elif "126.net" in url or "163.com" in url:
        headers["Referer"] = "https://music.163.com/"
    elif cfg.get("source", {}).get("cookie"):
        headers["Cookie"] = cfg["source"]["cookie"]

    suffix = ".mp3"
    lowered = url.lower()
    for ext in (".flac", ".m4a", ".m4s", ".wav", ".ogg", ".mp3"):
        if ext in lowered:
            suffix = ".m4a" if ext == ".m4s" else ext
            break
    target = dest.with_suffix(suffix)

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=not qq_media) as client:
                with client.stream("GET", url, headers=headers) as resp:
                    resp.raise_for_status()
                    ctype = (resp.headers.get("content-type") or "").lower()
                    if "text/html" in ctype or "application/json" in ctype:
                        raise SourceError(f"返回非音频内容 ({ctype})")
                    with target.open("wb") as handle:
                        for chunk in resp.iter_bytes(256 * 1024):
                            handle.write(chunk)
            last_exc = None
            break
        except (httpx.HTTPError, SourceError) as exc:
            last_exc = exc
            target.unlink(missing_ok=True)
            if attempt == 0 and not qq_media:
                logger.debug("下载重试 %s: %s", url[:60], exc)
    if last_exc is not None:
        if qq_media:
            raise SourceError("QQ 授权媒体下载失败") from None
        raise SourceError(f"音频下载失败: {last_exc}") from last_exc

    if not target.exists() or target.stat().st_size < 64 * 1024:
        target.unlink(missing_ok=True)
        raise SourceError("下载的文件过小，可能是无效链接或需要 VIP")
    logger.info("下载完成 %s (%.1f MB)", target.name, target.stat().st_size / 1024 / 1024)
    return target


def _try_download_candidate(
    candidate: dict[str, Any], cfg: dict, workdir: Path, tag: str = "input"
) -> tuple[Path, dict[str, Any]] | None:
    """尝试为单个候选下载完整音频；智能识别 30s VIP 试听片段。"""
    # B 站候选：支持主链 + backupUrl 多 CDN 容灾
    if candidate.get("bvid"):
        urls = _bili_audio_urls(cfg, candidate["bvid"])
        for u in urls:
            try:
                path = download(u, workdir / tag, cfg)
                size = path.stat().st_size
                if size < 0.35 * 1024 * 1024:
                    path.unlink(missing_ok=True)
                    continue
                return path, {
                    "title": candidate.get("title") or candidate["bvid"],
                    "artist": candidate.get("artist") or "",
                    "album": "",
                    "origin": "bilibili",
                    "origin_label": "B站",
                    "bvid": candidate["bvid"],
                    "size_mb": round(size / 1048576, 1),
                }
            except SourceError as exc:
                logger.debug("B 站镜像下载失败: %s", exc)
        return None

    url, origin, cover = best_url(cfg, candidate)
    if not url:
        return None
    try:
        path = download(url, workdir / tag, cfg)
    except SourceError as exc:
        logger.warning("候选下载失败 %s: %s", candidate.get("title"), exc)
        return None

    size = path.stat().st_size
    # 文件 < 2.0 MB 时用 ffprobe 检查真实时长，避免把 128k 完整短歌误杀，同时拦截 30s VIP 试听
    if size < 2.0 * 1024 * 1024:
        dur = _probe_audio_duration(path, cfg)
        if dur > 0 and dur < 55.0:
            logger.warning("候选 %s 仅 %.0fs（%.1f MB，判定为 VIP 试听），跳过",
                           candidate.get("title"), dur, size / 1048576)
            path.unlink(missing_ok=True)
            return None
        if dur <= 0 and size < 1.2 * 1024 * 1024:
            logger.warning("候选 %s 仅 %.1f MB 且时长未知，跳过",
                           candidate.get("title"), size / 1048576)
            path.unlink(missing_ok=True)
            return None

    info: dict[str, Any] = {
        "title": candidate.get("title") or "",
        "artist": candidate.get("artist") or "",
        "album": candidate.get("album") or "",
        "origin": origin,
        "origin_label": _ORIGIN_LABELS.get(origin, origin),
        "size_mb": round(size / 1048576, 1),
    }
    if cover:
        info["suspect_cover"] = True
        info["cover_note"] = "原唱版本未找到，已使用翻唱音频"
    return path, info


def _resolve_qq_recording(selected: dict, cookie: str, cfg: dict, workdir: Path) -> tuple[Path, dict]:
    mid = selected.get("songmid")
    if not mid:
        raise SourceError("所选 QQ 录音缺少 songmid，请重新选择")
    try:
        with httpx.Client(headers={"User-Agent": UA}) as client:
            detail, rows = qq_recording.lookup(client, cookie, mid)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        raise SourceError("所选 QQ 录音授权状态未知，请检查账号或稍后重试") from None

    try:
        expected = float(detail.get("interval") or 0)
    except (TypeError, ValueError):
        expected = 0
    reasons = []
    for row in rows:
        code = row.get("result")
        if code == 104003:
            reason = "104003（此音质未获授权）"
        elif type(code) is not int:
            reason = "授权状态未知"
        elif code != 0:
            reason = f"其它返回码 {code}"
        elif not row.get("url"):
            reason = "返回码 0，媒体地址为空"
        elif row.get("trial") is True:
            reason = "仅提供试听片段"
        elif not 0 < expected < float("inf"):
            reason = "官方时长未知，无法核验完整录音"
        else:
            try:
                path = download(row["url"], workdir / "input", cfg, qq_media=True)
            except SourceError:
                reason = "授权媒体下载失败"
            else:
                actual = _probe_audio_duration(path, cfg)
                if 0 < actual < float("inf") and abs(actual - expected) <= min(3.0, expected * 0.05):
                    return path, {"title": detail.get("title") or selected.get("title", ""),
                                  "artist": selected.get("artist", ""), "album": selected.get("album", ""),
                                  "origin": "qq", "origin_label": "QQ音乐", "songmid": mid,
                                  "duration": actual, "quality": row.get("quality"), "qq_recording_locked": True,
                                  "size_mb": round(path.stat().st_size / 1048576, 1)}
                path.unlink(missing_ok=True)
                reason = "下载时长与官方时长不符或无法核验"
        reasons.append(f"{row.get('quality') or '未知音质'}：{reason}")
    raise SourceError("所选 QQ 录音无可用完整版：" + ("；".join(reasons) or "授权状态未知"))


def resolve(spec: dict, cfg: dict, workdir: Path) -> tuple[Path, dict[str, Any]]:
    """把任务描述解析成完整音频文件。

    spec 支持：
      {"audio_path": "/abs/x.mp3"}      本地文件
      {"audio_url": "https://..."}      直链
      {"bvid": "BV..."}                 指定 B 站视频
      {"selected_track": {...}}         WebUI 搜选的指定候选
      {"song": "大鱼 周深"}            关键词 / BV 号 / 音频链接
    """
    workdir.mkdir(parents=True, exist_ok=True)

    local = spec.get("audio_path")
    if local:
        path = Path(local).expanduser()
        if not path.is_file():
            raise SourceError(f"本地音频不存在: {path}")
        if spec.get("upload_id"):
            return path, {"title": spec.get("song") or "上传音频", "artist": "", "origin": "upload", "origin_label": "上传"}
        return path, {"title": path.stem, "artist": "", "origin": "local", "origin_label": "本地"}

    direct = spec.get("audio_url")
    if direct:
        path = download(direct, workdir / "input", cfg)
        return path, {"title": spec.get("song") or path.stem, "artist": "", "origin": "url", "origin_label": "直链"}

    # WebUI 点选的指定候选
    selected = spec.get("selected_track")
    if ((cfg.get("source") or {}).get("qq_exact_recording") is True
            and isinstance(selected, dict) and selected.get("origin") == "qq"
            and int(spec.get("preview_seconds") or 0) <= 0):
        cookie = _qq_cookie(cfg)
        if cookie:
            return _resolve_qq_recording(selected, cookie, cfg, workdir)
    if isinstance(selected, dict) and (selected.get("bvid") or selected.get("gd_id") or selected.get("songmid") or selected.get("url")):
        res = _try_download_candidate(selected, cfg, workdir)
        if res is not None:
            return res
        logger.warning("指定候选下载失败，回退常规搜索: %s", selected.get("title"))

    # 指定 BV 号
    bvid = (spec.get("bvid") or "").strip()
    if bvid:
        track = _bili_track_from_bvid(cfg, bvid) or {"title": spec.get("song") or bvid, "artist": "", "bvid": bvid}
        res = _try_download_candidate(track, cfg, workdir)
        if res is not None:
            return res
        raise SourceError(f"B 站视频 {bvid} 音频提取失败")

    keyword = (spec.get("song") or "").strip()
    if not keyword:
        raise SourceError("缺少 song / audio_url / audio_path / bvid")

    # 直接粘贴了音频直链
    if _DIRECT_AUDIO_RE.match(keyword):
        path = download(keyword, workdir / "input", cfg)
        return path, {"title": path.stem, "artist": "", "origin": "url", "origin_label": "直链"}

    # 直接粘贴了 BV 号或 B 站链接
    inline_bv = _extract_bvid(keyword)
    if inline_bv:
        track = _bili_track_from_bvid(cfg, inline_bv) or {"title": inline_bv, "artist": "", "bvid": inline_bv}
        res = _try_download_candidate(track, cfg, workdir)
        if res is not None:
            return res
        raise SourceError(f"B 站视频 {inline_bv} 音频提取失败")

    source_cfg = cfg.get("source", {})
    limit = int(source_cfg.get("search_limit", 8))
    source_hint = str(spec.get("source_hint") or "auto")
    tracks = search(cfg, keyword, limit=limit, source_hint=source_hint)

    idx = int(spec.get("candidate_index") or 0)
    if idx < 0 or idx >= len(tracks):
        idx = 0
    tracks = tracks[idx:] + tracks[:idx]

    cleaned = _clean_keyword(keyword)
    tokens = [t for t in cleaned.split(" ") if t]
    tried_bili = any(t.get("origin") == "bilibili" for t in tracks)

    # 依次尝试前 5 个相关候选（跳过歌名完全无关的候选，避免下载错歌）
    for candidate in tracks[: max(3, min(limit, 5))]:
        if candidate.get("origin") != "bilibili":
            rel = _title_relevance(candidate.get("title", ""), candidate.get("artist", ""), cleaned, tokens)
            if rel < 0.2 and len(tracks) > 1:
                logger.info("跳过歌名不相关候选：%s - %s (rel=%.2f)", candidate.get("title"), candidate.get("artist"), rel)
                continue
        res = _try_download_candidate(candidate, cfg, workdir)
        if res is not None:
            path, info = res
            info["candidates"] = len(tracks)
            return path, info

    # 全音乐平台都没拿到完整版 → 自动启动 B 站全能救场！
    if not tried_bili and source_cfg.get("bili_enabled", True) and source_cfg.get("bili_fallback", True) and source_hint != "music":
        logger.info("音乐平台未获取到完整音频，启动 B 站自动救场：《%s》", keyword)
        try:
            bili_tracks = _search_bilibili(cfg, cleaned, limit=5, general_music=True)
            for candidate in bili_tracks[:3]:
                res = _try_download_candidate(candidate, cfg, workdir)
                if res is not None:
                    path, info = res
                    info["candidates"] = len(bili_tracks)
                    info["bili_fallback"] = True
                    return path, info
        except Exception as exc:  # noqa: BLE001
            logger.warning("B 站救场失败: %s", exc)

    raise SourceError(f"没找到《{keyword}》的完整音频，可以尝试加上歌手名或直接粘贴 B 站 BV 号")


def resolve_bili_solo_cover(keyword: str, cfg: dict, workdir: Path) -> tuple[Path, dict[str, Any]] | None:
    """当音乐平台音频干声严重劣化/疑似鬼畜或劣质翻录时，静默尝试从 B 站检索更干净的单人翻唱版本。

    找不到或下载失败时返回 None（绝不抛错、绝不拒单，调用方回退原音频继续唱）。
    """
    source_cfg = cfg.get("source", {})
    if not source_cfg.get("bili_enabled", True):
        return None
    cleaned = _clean_keyword(keyword)
    if not cleaned:
        return None

    solo_good = ("翻唱", "独唱", "cover", "完整版", "纯享", "女声", "男声", "清唱")
    solo_bad = ("鬼畜", "合唱", "多人", "串烧", "合集", "唢呐", "电音", "dj", "remix", "纯音乐", "伴奏", "教学", "reaction")

    candidates: list[dict[str, Any]] = []
    seen_bv: set[str] = set()
    for q in (f"{cleaned} 翻唱", cleaned):
        try:
            found = _search_bilibili(cfg, q, limit=6, general_music=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("B 站单人翻唱检索失败 (%s): %s", q, exc)
            continue
        for item in found:
            bv = item.get("bvid") or ""
            if not bv or bv in seen_bv:
                continue
            seen_bv.add(bv)
            candidates.append(item)
        if len(candidates) >= 6:
            break

    if not candidates:
        return None

    def _solo_score(item: dict[str, Any]) -> float:
        t = f"{item.get('title', '')} {item.get('artist', '')}".lower()
        sc = 0.0
        for kw in solo_good:
            if kw in t:
                sc += 2.0
        for kw in solo_bad:
            if kw in t:
                sc -= 4.5
        dur = int(item.get("duration") or 0)
        if 120 <= dur <= 340:
            sc += 1.5
        elif dur > 0 and dur < 75:
            sc -= 3.0
        return sc

    candidates.sort(key=_solo_score, reverse=True)
    for cand in candidates[:3]:
        if _solo_score(cand) < -2.0:
            continue
        res = _try_download_candidate(cand, cfg, workdir, tag="bili_solo")
        if res is not None:
            path, info = res
            info["bili_solo_rescue"] = True
            return path, info
    return None
