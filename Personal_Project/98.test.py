from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
from selenium.webdriver.chrome.options import Options
import time
import urllib.request # <- 이미지 url 주소로 다운받을 때 필요

# op = Options()
# op.add_experimental_option('prefs',{'download.default_directiory':r'd:/study_data/_data/image/PP/'})
# urllib.request.urlretrieve(imgUrl, 'filePath' + 'fileName' + ".fileForm")

keyword = 'mark43'
chromedriver = 'c:/study/chromedriver.exe' # 크롬드라이버 파일을 놔둔 경로 작성 필요 
driver = webdriver.Chrome(chromedriver) 
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl") # 구글 이미지 검색 url
elem = driver.find_element_by_name("q") # 구글 검색창 선택
elem.send_keys(keyword) # 검색창에 검색할 내용(name)넣기
elem.send_keys(Keys.RETURN) # enter 입력

img_folder = f"d:/study_data/_data/image/PP/{keyword}" # 원하는 경로에 키워드 이름으로 폴더 자동생성
 
# img_folder = './img'

    

images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") # 이미지의 클래스 이름을 images 라는 변수에 담아주고 
result = []

if not os.path.isdir(img_folder) : # 없으면 새로 생성하는 조건문 
    os.mkdir(img_folder)


# for image in images: # 이미지들 중에 각각 개별 이미지를 하나씩 뽑아서
#     image.click() # 그 이미지를 클릭하도록 해줌
#     time.sleep(2.5) # 로딩시간 감안해서 3초 대기하고
#     # imgUrl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src") # url을 찾아서 





for index, link in enumerate(result) :
    start = link.rfind('.')
    end = link.rfind('?')
    filetype = link[start:end]
    urllib.request.urlretrieve(link, './imgs/{}{}'.format(index, filetype))

time.sleep(3)