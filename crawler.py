from time import sleep

from tqdm import tqdm

from browser import Browser



class InstaCrawler():
    URL = "https://www.instagram.com"

    def __init__(self, has_screen=False):
        self.browser = Browser(has_screen)
        self.login()

    def login(self):
        browser = self.browser
        url = "%s/accounts/login/" % (InstaCrawler.URL)
        browser.get(url)
        u_input = browser.find_one('input[name="username"]')
        u_input.send_keys('01044062835')
        p_input = browser.find_one('input[name="password"]')
        p_input.send_keys('vancouver@1136')

        login_btn = browser.find_one(".L3NKy")
        login_btn.click()

    def get_posts_tag(self, tag, num):
        url = "%s/explore/tags/%s/" % (InstaCrawler.URL, tag)
        self.browser.get(url)

        TIMEOUT = 600
        browser = self.browser
        key_set = set()
        posts = []
        pre_post_num = 0
        wait_time = 1

        progressbar = tqdm(total=num)

        def start_fetching(pre_post_num, wait_time):
            element_posts = browser.find(".v1Nh3 a")
            for element in element_posts:
                key = element.get_attribute("href")
                if key not in key_set:
                    dict_post = {"key": key}
                    element_img = browser.find_one(".KL4Bh img", element)
                    dict_post["caption"] = element_img.get_attribute("alt")
                    dict_post["img_url"] = element_img.get_attribute("src")

                    key_set.add(key)
                    posts.append(dict_post)

                    if len(posts) == num:
                        break

            if pre_post_num == len(posts):
                progressbar.set_description("Wait for %s sec" % (wait_time))
                sleep(wait_time)
                progressbar.set_description("fetching")

                wait_time *= 2
                browser.scroll_up(300)
            else:
                wait_time = 1

            pre_post_num = len(posts)
            browser.scroll_down()

            return pre_post_num, wait_time

        progressbar.set_description("fetching")
        while len(posts) < num and wait_time < TIMEOUT:
            post_num, wait_time = start_fetching(pre_post_num, wait_time)
            progressbar.update(post_num - pre_post_num)
            pre_post_num = post_num

            loading = browser.find_one(".W1Bne")
            if not loading and wait_time > TIMEOUT / 2:
                break

        progressbar.close()
        print("Done. Fetched %s posts." % len(posts))
        return posts[:num]
