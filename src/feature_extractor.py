import re
import ipaddress

from urllib.parse import urlparse


class FeatureExtractor:

    def url_length(self, url):
        return len(url)

    def digit_count(self, url):
        return sum(
            c.isdigit()
            for c in url
        )

    def special_char_count(self, url):

        return len(
            re.findall(
                r'[@%?=_\-]',
                url
            )
        )

    def dot_count(self, url):
        return url.count(".")

    def has_https(self, url):

        return int(
            url.startswith("https")
        )

    def has_ip(self, url):

        try:

            domain = (
                urlparse(url)
                .netloc
                .split(":")[0]
            )

            ipaddress.ip_address(domain)

            return 1

        except:

            return 0

    def subdomain_count(self, url):

        domain = (
            urlparse(url)
            .netloc
        )

        return max(
            0,
            len(domain.split(".")) - 2
        )

    def hyphen_count(self, url):

        return url.count("-")

    def extract(self, url):

        return {

            "url_length":
                self.url_length(url),

            "digit_count":
                self.digit_count(url),

            "special_char_count":
                self.special_char_count(url),

            "dot_count":
                self.dot_count(url),

            "https":
                self.has_https(url),

            "has_ip":
                self.has_ip(url),

            "subdomain_count":
                self.subdomain_count(url),

            "hyphen_count":
                self.hyphen_count(url)
        }