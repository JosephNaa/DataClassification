import os
from time import sleep
import random

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class Browser:
    def __init__(self, has_screen):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        service_args = ["--ignore-ssl-errors=true"]
        chrome_options = Options()
        if not has_screen:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--no-sandbox")
        self.driver = webdriver.Chrome(
            executable_path="%s/bin/chromedriver" % dir_path,
            service_args=service_args,
            chrome_options=chrome_options,
        )
        self.driver.implicitly_wait(5)

    def get(self, url):
        self.driver.get(url)

    def find_one(self, css_selector, elem=None, waittime=0):
        obj = elem or self.driver

        if waittime:
            WebDriverWait(obj, waittime).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
            )

        try:
            return obj.find_element(By.CSS_SELECTOR, css_selector)
        except NoSuchElementException:
            return None

    def find(self, css_selector, elem=None, waittime=0):
        obj = elem or self.driver

        try:
            if waittime:
                WebDriverWait(obj, waittime).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
                )
        except TimeoutException:
            return None

        try:
            return obj.find_elements(By.CSS_SELECTOR, css_selector)
        except NoSuchElementException:
            return None

    def scroll_down(self, wait=0.3):
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        rand_sleep(wait)

    def scroll_up(self, offset=-1, wait=2):
        if offset == -1:
            self.driver.execute_script("window.scrollTo(0, 0)")
        else:
            self.driver.execute_script("window.scrollBy(0, -%s)" % offset)
        rand_sleep(wait)

    def __del__(self):
        self.driver.quit()

def rand_sleep(average=1):
    _min, _max = average * 1 / 2, average * 3 / 2
    sleep(random.uniform(_min, _max))