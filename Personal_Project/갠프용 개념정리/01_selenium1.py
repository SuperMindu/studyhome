from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import os
import urllib.request


# 이미지들이 createFolder를 지정하여 이미지들이 저장될 폴더를 만들어지게 함
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error : Creating directory.' + directory)
        

# 검색어 입력 및 폴더 생성
keyword = 'mark43'
createFolder('d:/study_data/_data/image/PP/' + keyword + '_img_download')

chromedriver = 'c:/study/chromedriver.exe'
driver = webdriver.Chrome(chromedriver)
driver.implicitly_wait(1)

print(keyword, 'searching...')
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl") # 구글 이미지 검색 url
elem = driver.find_element_by_name("q") # 구글 검색창 선택
elem.send_keys(keyword) # 검색창에 검색할 내용(name)넣기
elem.send_keys(Keys.RETURN) # enter 입력


# 스크롤다운. '결과 더보기' 버튼 나오면 클릭해서 계속 스크롤 다운
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

# 검색해서 나온 이미지 개수 종합
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
print(keyword + ' number of image : ', len(images))
time.sleep(3)

# 원본 이미지 다운로드
links = []
for i in range(1,len(images)):
    try:
        driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').click()
        links.append(driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img').get_attribute('src'))
        driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[2]/a').click()
        print(keyword+' 링크 수집 중..... number :'+str(i)+'/'+str(len(images)))
    except:
        continue

forbidden=0
for k,i in enumerate(links):
    try:
        url = i
        start = time.time()
        urllib.request.urlretrieve(url, "d:/study_data/_data/image/PP/" + keyword + "_high resolution/" + keyword + "_" + str(k-forbidden) + ".jpg")
        print(str(k+1)+'/'+str(len(links))+' '+keyword+' 다운로드 중....... Download time : '+str(time.time() - start)[:5]+' 초')
    except:
        forbidden+=1
        continue

print(keyword + ' image download completed')

driver.close()

'''
# 이미지 개수
links=[]
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
for image in images:
    if image.get_attribute('src')!=None:
        links.append(image.get_attribute('src'))

print(keyword+' 찾은 이미지 개수:',len(links))
time.sleep(2)

# 이미지 다운로드
for k,i in enumerate(links):
    url = i
    start = time.time()
    urllib.request.urlretrieve(url, "./"+keyword+"_img_download/"+keyword+"_"+str(k)+".jpg")
    print(str(k+1)+'/'+str(len(links))+' '+keyword+' 다운로드 중....... Download time : '+str(time.time() - start)[:5]+' 초')
print(keyword+' ---다운로드 완료---')

driver.close()
'''

