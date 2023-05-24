import requests
import ssl
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import urlparse
from urllib.parse import urljoin

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def parseurl(url):
    document = urlopen(url, context=ctx)
    html = document.read()
    if 'text/html' != document.info().get_content_type() :
        print("Ignore non text/html page")


    soup = BeautifulSoup(html, "html.parser")
    # Retrieve all of the anchor tags
    tags = soup('a')
    for tag in tags:
        href = tag.get('href', None)
        if ( href is None ) : continue
        # Resolve relative references like href="/contact"
        up = urlparse(href)
        if ( len(up.scheme) < 1 ) :
            href = urljoin(url, href)
        ipos = href.find('#')
        if ( ipos > 1 ) : href = href[:ipos]
        if ( href.endswith('.png') or href.endswith('.jpg') or href.endswith('.gif') ) : continue
        if ( href.endswith('/') ) : 
            href = href[:-1]
            # print href
            print(href)
        if ( len(href) < 1 ) : continue

starturl = 'https://le.ac.uk/informatics/'
parseurl(starturl)