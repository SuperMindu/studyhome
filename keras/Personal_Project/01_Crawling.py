from bs4 import BeautifulSoup # HTML 및 XML 문서를 구문 분석 하기위한 Python의 패키지
from urllib.request import urlopen

with urlopen('https://en.wikipedia.org/wiki/Main_Page') as response:
    soup = BeautifulSoup(response, 'html.parser')
    for anchor in soup.find_all('a'):
        print(anchor.get('href', '/'))