# https://www.youtube.com/watch?v=1b7pXC1-IbE <--- 유튜브 참고 
# https://velog.io/@jungeun-dev/Python-%EC%9B%B9-%ED%81%AC%EB%A1%A4%EB%A7%81Selenium-%EA%B5%AC%EA%B8%80-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%88%98%EC%A7%91 <-- 
# https://yobbicorgi.tistory.com/29 < --- 여기가 찐또배기



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

# driver = webdriver.Chrome() # 크롬드라이버 설치한 경로 작성 필요 
# driver.get("https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl") # 구글 이미지 검색 url
# elem = driver.find_element_by_name("q") # 구글 검색창 선택
# elem.send_keys("iron man") # 검색창에 검색할 내용(name)넣기
# elem.send_keys(Keys.RETURN) # enter 입력

'''
driver.find_elements_by_css_selector(".rg_i.Q4LuWd")[0].click() # <- .click() 은 말 그대로 클릭 하겠다는 뜻
# element는 하나만 elements는 여러개 선택할 때
# 선택하고자 하는 영역의 class 이름으로 찾기. class라서 띄어쓰기 없이 . 으로 연결하는거임. 맨 앞에도 . 을 넣어서 클래스 라는 걸 정의해줘야 됨. 
# find_elements 는 검색해서 나온 여러개의 작은 이미지들 여러개를 하나씩 끄집어 내서 클릭을 해야하기 때문에 
# 가장 첫번째 요소를 클릭을 해서 그거를 클릭하겠다 그래서 마지막에 [0]을 해주고 .click()을 해줌.

time.sleep(3) # <- 만약에 크롤링 하다가 중간중간에 '시스템에 부착된 장치가 작동하지 않습니다.' 이런 메세지가 뜨면 웹페이지의 로딩은 되지도 않았는데 코드만 너무 빨리 실행됐을 확률이 높으므로
# 시간이 좀 필요할 것 같은 위치에 time.sleep 넣어줘서 잠깐 대기시간을 가지게 만드는 것도 방법임

############## 중요 ################
# 위에서처럼 실행해서 이미지를 클릭해서 크게 만들면 그 이미지의 클래스는 "n3VNCb"
# 그래서 그 이미지를 다운 받으려면 class 뒤에 있는 src주소를 알아야 함
imgUrl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src") # <- 이게 그 class로 선택을 하고 해당 이미지의 src 주소를 가져오는 코드

urllib.request.urlretrieve(imgUrl, "test.jpg") # <- 이미지 url 주소로 다운 받는 코드 

# 일단 이렇게만 하면 끝인데 우리는 이 작업을 반복해서 돌리면서 이미지 데이터를 수집해야함
# 반복 작업이 필요할 떈 뭐다? ㅎㅎ
'''

# SCROLL_PAUSE_TIME = 1.5

# # Get scroll height
# last_height = driver.execute_script("return document.body.scrollHeight") 
# # execute_script는 JavaScript코드를 실행하는 코드임. 그리고 괄호 안의 JavaScript코드는 브라우저의 높이를 알 수 있는 코드임
# # 즉, 브라우저의 높이를 알 수 있는 자바코드를 실행시켜서 그 값을 last_height 이라는 변수에 저장해줌

# while True: # 무한반복
#     # Scroll down to bottom
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # 대충 브라우저의 끝까지 스크롤을 내리겠다는 자바 코드임

#     # Wait to load page
#     time.sleep(SCROLL_PAUSE_TIME) # 로드 될때 동안 SCROLL_PAUSE_TIME 값 만큼 기다려주겠다는 뜻

#     # Calculate new scroll height and compare with last scroll height
#     new_height = driver.execute_script("return document.body.scrollHeight") # 로딩이 끝나고 나면 이번엔 브라우저의 높이를 다시 구해서 new_height 라는 변수에 담아줌
#     if new_height == last_height: # 만약에 새로 구한 높이랑 이전 높이랑 같다면 (스크롤을 내렸을 때 더 나오는 게 없다면)
#         try:
#             driver.find_element_by_css_selector(".mye4qd").click() # <- '결과 더보기' 버튼이 나오면 클릭해라 
#         except:
#             break
#         # break # 무한루프를 빠져나오겠다
#     last_height = new_height
# 하지만 구글 이미지 로딩창 에서는 맨 마지막에 스크롤은 더 안 내려가는데 '결과 더보기' 버튼이 있는 경우가 있어서 클릭을 하면 이미지가 더 나오는 경우가 있음
# 그래서 그 버튼을 클릭하는 코드를 while문 안에 넣어줘야 함

# 그러나~ 이렇게 실행을 하게 되면 잘 되긴 하는데 로딩을 끝까지 했을 때 마지막에 결과 더보기 버튼이 없어지기 때문에 잘 되긴 하는데 종료가 될때 오류가 뜨면서 종료됨
# 그래서 파이썬의 try: except: 라는 걸 써서 try: except: 사이에 있는 코드를 실행을 했는데 
# 더이상 실행을 하지 못하는 상황에 맞닥뜨렸을 때 (오류가 났을 때) 밑에 break로 빠져나오면서 무한루프를 빠져나오게 만들어줌
# 이렇게 해야 비로소 아래의 이미지 다운 코드를 실행시킬 수 있음
# 들여쓰기에 유의하자


images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") # 이미지의 클래스 이름을 images 라는 변수에 담아주고 
count = 1
for image in images: # 이미지들 중에 각각 개별 이미지를 하나씩 뽑아서
    image.click() # 그 이미지를 클릭하도록 해줌
    time.sleep(2.5) # 로딩시간 감안해서 3초 대기하고
    imgUrl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src") # url을 찾아서 
    urllib.request.urlretrieve(imgUrl, 'd:/PP/' + str(count) + ".jpg") # <- 이게 다운받을 폴더명 지정해서 받음
    # urllib.request.urlretrieve(imgUrl, str(count) + ".jpg") # 다운 받도록 만들면 됨 
    count = count + 1
# 근데 우리는 다운받은 이미지들의 이름을 똑같이 저장하면 안되니까 위에 count라는 변수를 지정해주고 그 값을 1로 줘서 다운을 받을 때마다 파일 이름 숫자가 1씩 늘어나면서 저장되게끔 해줌
# 이 count로 이름을 지정해줘야 함 근데 count는 숫자형이기 때문에 문자형끼리 더해질 수 있게 str()로 감싸줘서 string형 자료형으로 만들어줌
# 이렇게 하면 50장 밖에 저장이 안됨. 왜? why? 처음에 이미지를 검색을 했을때 나오는 이미지의 개수가 50개이기 때문에
# 그래서 사람은 스크롤을 내려가면서 이미지를 더 로드해서 추가적으로 찾을 수 있음. 심지어 계속 내리다보면 '결과 더보기' 버튼을 클릭해야만 더 로드가 되는 지점이 있음
# 근데 컴터는 그걸 못하기 때문에 이걸 또 코딩 해줘야 함 
# 근데 이미지 다운받기 전에 미리 스크롤을 바닥까지 내려놓고 그다음에 다운을 받게 시키면 더 시간이 절약되지 않을까? 
# 그래서 검색을 마치고 난 직후에 스크롤다운 코드를 넣어줌 36번째 줄부터 참고 해봐


time.sleep(5) # n초의 시간동안 대기

# assert "Python" in driver.title
# elem = driver.find_element_by_name("q")
# elem.clear()
# elem.send_keys("pycon")
# elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
driver.close()

# 자 이걸 응용해서 내가 원하는 경로에 원하는 파일명을 생성해서 그 안에 저장해주는 코드도 추가 해주자


