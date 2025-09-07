import ipaddress
import socket
import re

PRIVATE_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]
METADATA_HOSTS = {"169.254.169.254", "metadata", "metadata.google.internal"}

def _resolve(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None)
    return [ai[4][0] for ai in infos]

def _is_private(ip: str) -> bool:
    ipaddr = ipaddress.ip_address(ip)
    return any(ipaddr in net for net in PRIVATE_NETS)

def is_safe_url(url: str) -> bool:
    m = re.match(r"^(https?)://([^/:]+)(:\d+)?(/|$)", url, re.I)
    if not m: return False
    scheme, host = m.group(1).lower(), m.group(2).lower()
    if scheme != "https": return False
    if host in METADATA_HOSTS: return False
    for ip in _resolve(host):
        if _is_private(ip): return False
    return True