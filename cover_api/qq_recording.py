"""Resolve an exact QQ recording. Never forward account headers to a media host."""
import re
from urllib.parse import urlsplit

API = 'https://u.y.qq.com/cgi-bin/musicu.fcg'


def account_id(cookie):
    fields = dict(part.strip().partition('=')[::2] for part in cookie.split(';'))
    for key in ('musicid', 'qqmusic_uin', 'wxuin', 'uin'):
        value = fields.get(key, '').removeprefix('o')
        if re.fullmatch(r'[1-9][0-9]{0,19}', value):
            return value
    return '0'


def lookup(client, cookie, songmid):
    if not re.fullmatch(r'[A-Za-z0-9]{1,32}', songmid):
        raise ValueError('QQ recording ID invalid')
    if cookie and account_id(cookie) == '0':
        raise ValueError('QQ account ID missing from configured session')
    headers = {'Referer': 'https://y.qq.com/'}
    if cookie:
        headers['Cookie'] = cookie
    uin = account_id(cookie)

    def rpc(module, method, param):
        body = {'comm': {'uin':uin, 'ct':24, 'cv':0, 'format':'json'}, 'req_0':{'module':module, 'method':method, 'param':param}}
        response = client.post(API, json=body, headers=headers, follow_redirects=False, timeout=12)
        response.raise_for_status()
        envelope = response.json()
        packet = envelope.get('req_0') or {}
        if envelope.get('code') != 0 or packet.get('code') != 0:
            raise ValueError('QQ lookup rejected')
        return packet['data']

    track = rpc('music.pf_song_detail_svr', 'get_song_detail_yqq', {'song_mid':songmid})['track_info']
    if track.get('mid') != songmid:
        raise ValueError('QQ recording mismatch')
    media = track['file']['media_mid']
    if not re.fullmatch(r'[A-Za-z0-9]{1,32}', media):
        raise ValueError('QQ media ID invalid')
    files = {f'F000{media}.flac':'FLAC', f'M800{media}.mp3':'MP3-320', f'M500{media}.mp3':'MP3-128'}
    data = rpc('vkey.GetVkeyServer','CgiGetVkey',{'guid':'10000','songmid':[songmid]*len(files),'songtype':[0]*len(files),'uin':uin,'loginflag':int(bool(cookie)),'platform':'20','filename':list(files)})
    by_name = {r.get('filename'):r for r in data.get('midurlinfo',[]) if isinstance(r,dict)}
    rows = []
    for filename, quality in files.items():
        r = by_name.get(filename, {})
        path = r.get('purl') or ''
        url = ''
        if r.get('result') == 0 and path and not path.startswith('/') and not urlsplit(path).scheme and not any(c in path for c in '\\\r\n'):
            url = 'https://ws.stream.qqmusic.qq.com/' + path
        rows.append({'quality':quality,'result':r.get('result'),'url':url,'trial':bool(r['freeTrialInfo']) if 'freeTrialInfo' in r else None})
    return track, rows
