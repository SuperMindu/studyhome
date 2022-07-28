from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import urllib.request


# def createDirectory(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print("Error: Failed to create the directory.")

keyword = 'mark43'
img_number = 100

# img_folder = f"d:/study_data/_data/image/PP/{keyword}"
# if not os.path.isdir(img_folder) : # 없으면 새로 생성하는 조건문 
#     os.mkdir(img_folder)


# chromedriver = 'c:/study/chromedriver.exe' # 크롬드라이버 파일을 놔둔 경로 작성 필요 
# driver = webdriver.Chrome(chromedriver) 
# driver.get("https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl") # 구글 이미지 검색 url
# elem = driver.find_element_by_name("q") # 구글 검색창 선택
# elem.send_keys(keyword) # 검색창에 검색할 내용(name)넣기
# elem.send_keys(Keys.RETURN)
# chromedriver = 'c:/study/chromedriver.exe' # 크롬드라이버 파일을 놔둔 경로 작성 필요 
# driver = webdriver.Chrome(chromedriver)

def crawling_img(keyword):
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install())) #드라이버 위치 자동으로 찾아줌
    driver.get("https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl")
    elem = driver.find_element_by_name("q")
    elem.send_keys(keyword)
    elem.send_keys(Keys.RETURN)
    
    
    SCROLL_PAUSE_TIME = 1
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")  # 브라우저의 높이를 자바스크립트로 찾음
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 브라우저 끝까지 스크롤을 내림
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_elements_by_css_selector(".mye4qd").click()
            except:
                break
        last_height = new_height

    imgs = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
    dir = "D:/PP"+ "/" + keyword
    if not os.path.isdir(dir) : # 폴더가 없으면 새로 생성하는 조건문 
        os.mkdir(dir)
    
    # createDirectory(dir) #폴더 생성해준다
    count = 1
    for img in imgs:
        try:
            img.click()
            time.sleep(2)
            imgUrl = driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")
            path = "D:/PP/" + keyword + "/"
            urllib.request.urlretrieve(imgUrl, path + "img_" + str(count) + ".jpg")
            count = count + 1
            if count >= img_number: #이미지 장수 선택 
                break
        except:
            print("안되는디") #경로못찾으면 패~쓰~~~~~
    driver.close()
    
    
    




# images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") # 이미지의 클래스 이름을 images 라는 변수에 담아주고 
# count = 1
# for image in images: # 이미지들 중에 각각 개별 이미지를 하나씩 뽑아서
#     image.click() # 그 이미지를 클릭하도록 해줌
#     time.sleep(2.5) # 로딩시간 감안해서 3초 대기하고
#     imgUrl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src") # url을 찾아서 
#     urllib.request.urlretrieve(imgUrl, "d:/study_data/_data/image/PP/" + str(count) + ".jpg") # <- 이게 다운받을 폴더명 지정해서 받음
#     # urllib.request.urlretrieve(imgUrl, str(count) + ".jpg") # 다운 받도록 만들면 됨 
#     count = count + 1
    



