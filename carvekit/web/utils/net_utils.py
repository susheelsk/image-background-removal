import socket
import struct
from urllib.parse import urlparse


def is_loopback(address):
    host = None

    try:
        parsed_url = urlparse(address)
        host = parsed_url.hostname
    except:
        return False# url is not even a url

    loopback_checker = {
        socket.AF_INET: lambda x: struct.unpack('!I', socket.inet_aton(x))[0] >> (32-8) == 127,
        socket.AF_INET6: lambda x: x == '::1'
    }
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            r = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
        except socket.gaierror:
            return False
        for family, _, _, _, sockaddr in r:
            if not loopback_checker[family](sockaddr[0]):
                return False
    return True