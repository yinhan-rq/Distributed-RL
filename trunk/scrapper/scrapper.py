from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import os
import json
'''
用于爬取kaggle竞赛上google football 的数据,并进行数据转换
在windows本机上带代理测试成功,服务器上存在网络问题
'''

# 文件保存地址
download_path = "./data"
agent_list = ['WeKick','SaltyFish','Raw Beast']
#agent名和提交号，用于组装爬取地址
agent_collect = {
    'WeKick':'18357617',
    'SaltyFish':'18391341',
    'Raw Beast':'18379505'
}

ch_options = webdriver.ChromeOptions()

# 避免SSL错误
ch_options.add_argument('--ignore-certificate-errors')
ch_options.add_argument('--ignore -ssl-errors')

# 为Chrome配置无头模式,
# 在服务器上跑需要打开，因为没有图形界面
# ch_options.add_argument("--headless")  
# ch_options.add_argument('--no-sandbox')
# ch_options.add_argument('--disable-gpu')
# ch_options.add_argument('--disable-dev-shm-usage')

# 设置浏览器偏好
prefs = {'profile.default_content_settings.popups': 0, #防止保存弹窗
'download.default_directory': download_path ,#设置默认下载路径
"profile.default_content_setting_values.automatic_downloads":1#允许多文件下载
}

# ch_options.add_experimental_option('prefs', prefs)
# # 在启动浏览器时加入配置
# dr = webdriver.Chrome(options=ch_options)

scrap_url = "https://www.kaggleusercontent.com/episodes/"
leaderboard_url = "https://www.kaggle.com/competitions/google-football/leaderboard?dialog=episodes-submission-"

for key,value in agent_collect:
    #添加偏好设置，在浏览器启动时设置下载地址，按agent分门别类(目前只能开多个浏览器，因为下载地址只找到初始化设置的方法)
    prefs['download.default_directory'] = download_path + '/' + str(key)
    ch_options.add_experimental_option('prefs', prefs)
    #启动浏览器
    dr = webdriver.Chrome(options=ch_options)
    url = leaderboard_url + value
    print("--------------------getting url---------------------")
    print("key:" + key + "," + "value:" + value)
    dr.get(url)

    # 动态网页，需要下拉操作才能加载全部项
    for i in range(400):
        dr.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        ActionChains(dr).key_down(Keys.DOWN).perform()

    # x_path
    id_str = dr.find_elements_by_xpath('//*[@id="site-content"]/div[3]/div[1]/div[1]/div/div[1]/ul/div/li[2]/a')
    collect_num = 0 

    # 遍历找到的x_path
    for i in id_str:
        temp = i.get_attribute('href') 
        episode_id = temp[-7:]
        json_url = scrap_url + episode_id + ".json"
        dr.get(json_url)
        collect_num += 1
        print("collect_num:" + str(collect_num))







