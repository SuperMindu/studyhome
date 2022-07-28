from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request 


# def createFolder(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print('Error : Creating directory.' + directory)

# 검색어 입력 및 폴더 생성
keyword = 'mark43'
# createFolder('d:/study_data/_data/image/PP/' + keyword + '_img_download')

chromedriver = 'c:/study/chromedriver.exe'
driver = webdriver.Chrome(chromedriver)
driver.implicitly_wait(1)

print(keyword, 'searching...')
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl") # 구글 이미지 검색 url
elem = driver.find_element_by_name("q") # 구글 검색창 선택
elem.send_keys(keyword) # 검색창에 검색할 내용(name)넣기
elem.send_keys(Keys.RETURN) # enter 입력

# 스크롤 다운
print(keyword + 'scrolling... please wait...')
elem1 = driver.find_element_by_tag_name("body")
for i in range(60):
    elem1.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1)
    
try:
    driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div[1]/div[2]/div[2]/input').click()
    
    for i in range(100):
        elem1.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.1)
except:
    pass

for i in range(500):
        elem1.send_keys(Keys.PAGE_UP)
        time.sleep(0.1)
# SCROLL_PAUSE_TIME = 1.5

# # Get scroll height
# last_height = driver.execute_script("return document.body.scrollHeight") 

# while True: 
#     # Scroll down to bottom
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") 

#     # Wait to load page
#     time.sleep(SCROLL_PAUSE_TIME) 

#     # Calculate new scroll height and compare with last scroll height
#     new_height = driver.execute_script("return document.body.scrollHeight") 
#     if new_height == last_height: 
#         try:
#             driver.find_element_by_css_selector(".mye4qd").click() 
#         except:
#             break
#         last_height = new_height

# 폴더생성
print("폴더생성")
img_folder = 'd:/study_data/_data/image/PP/'
 
if not os.path.isdir(img_folder) : # 없으면 새로 생성하는 조건문 
    os.mkdir(img_folder)


# 다운로드
# images = driver.find_elements(by=css, value=".rg_i.Q4LuWd")
images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") 
count = 1
for image in images: 
    image.click() 
    time.sleep(3) 
    imgUrl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src") 
    urllib.request.urlretrieve(imgUrl, str(count) + ".jpg")
    count = count + 1


time.sleep(5) 


# driver.close()



