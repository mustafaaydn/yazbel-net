# Forum YazBel'de kayıtlı bir kullanıcının şu ana dek yazdığı yanıtları
# elde edip aynı dizindeki `replies` klasörüne `{username}.txt` şeklinde
# kaydetmeyi sağlayan fonksiyonlar kümesini barındıran yer.
import logging
import os
import pathlib
import time

from selenium import webdriver
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.common.keys import Keys

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def go_to_replies_page(username, driver_path="chromedriver"):
    r"""
    Forumda kullanıcının verdiği cevapların olduğu kısma selenium yardımıyla
    gider.

    Parameters
    ------------
    username: str
        Kullanıcı adı örn. trdjango. Case-insensitive

    driver_path: str, optional, default="chromedriver"
        Google Chrome tarayıcısının driver'ının yani `chromedriver.exe`
        uygulamasının olduğu dizini söyler örn. r"C:\users\zz\chromedriver.exe"
        Eğer verilmediyse, PATH'te bulunduğunu varsayıyoruz

    Returns
    --------
    driver: selenium.webdriver.chrome.webdriver.WebDriver
        Kullanılan driver
    """
    driver = webdriver.Chrome(executable_path=driver_path)
    driver.get(f"https://forum.yazbel.com/u/{username}/activity/replies")
    return driver


def scroll_to_end_of_page(driver):
    """
    Taa en sona kadar iniyoruz sayfada, kaynak:
    https://stackoverflow.com/a/51345544/9332187 ve üstteki cevaplar felan.
    Halihazırda verilen driver'ın aktif olarak bir sayfada beklediğini
    varsayıyor örn. go_to_replies_page(username) sonrası çağrılabilir.

    Parameters
    -----------
    driver: selenium.webdriver.chrome.webdriver.WebDriver
        Kullanılan driver

    Returns
    ---------
    driver: selenium.webdriver.chrome.webdriver.WebDriver
        Kullanılan driver
    """
    # Get scroll height (nihayete erdirmek için scrollamayı)
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # to the end!
        html = driver.find_element_by_tag_name("html")
        html.send_keys(Keys.END)

        # Wait a bit to load the page
        time.sleep(1)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    return driver


def click_to_carets(driver):
    """
    "https://forum.yazbel.com/u/{username}/activity/replies" adreslerinde
    görüldüğü üzere, reply'ların bazları (uzun olanlar) kısa olarak sunuluyor
    ilk başta. Sağ üst köşede küçük ^ işareti var ona tıklarsak tüm yanıt
    gözüküyor. Bu fonksiyon bulduğu ^'lara tıklıyor.

    Parameters
    -----------
    driver: selenium.webdriver.chrome.webdriver.WebDriver
        Kullanılan driver

    Returns
    ---------
    driver: selenium.webdriver.chrome.webdriver.WebDriver
        Kullanılan driver
    """
    # expandable linkleri alalım ve tıklayalım
    expandable_links = driver.find_elements_by_class_name("expand-item")
    for link in expandable_links:
        try:
            link.click()
            time.sleep(1)
        except ElementClickInterceptedException:
            logging.warn("Failed to click a link")
    return driver


def write_replies(driver, username):
    """
    Cevapları alıyor ve "replies/username.txt"yi yazıyoruz ("w" kipinde).

    Parameters
    -----------
    driver: selenium.webdriver.chrome.webdriver.WebDriver
        Kullanılan driver

    username: str
        Kullanıcı adı örn. trdjango. Case-insensitive

    Returns
    --------
    driver: selenium.webdriver.chrome.webdriver.WebDriver
        Kullanılan driver
    """
    # replies klasörü oluşturalım yoksa
    pathlib.Path("replies").mkdir(exist_ok=True)

    # yanıtları alıp yazarız
    replies = driver.find_elements_by_class_name("excerpt")

    path_to_file = os.path.join("replies", f"{username}.txt")
    with open(path_to_file, "w", encoding="utf-8") as fh:
        for reply in replies:
            content = reply.get_attribute("innerText")
            fh.write(content)
            fh.write("\n"*2)

    return driver


def save_user_replies(username, driver_path="chromedriver"):
    R"""
    Yukarıdaki 4 fonksiyonu ardışık şekilde çağıran, parçaları bir araya
    getirip kullanıcının yanıtlarını ilgili dosyaya yazan fonksiyon.

    Parameters
    -----------
    username: str
        Kullanıcı adı örn. trdjango. Case-insensitive

    driver_path: str, optional, default="chromedriver"
        Google Chrome tarayıcısının driver'ının yani `chromedriver.exe`
        uygulamasının olduğu dizini söyler örn. r"C:\users\zz\chromedriver.exe"
        Eğer verilmediyse, PATH'te bulunduğunu varsayıyoruz

    Returns
    --------
    driver: selenium.webdriver.chrome.webdriver.WebDriver
        Kullanılan driver
    """
    driver = go_to_replies_page(username, driver_path)
    driver = scroll_to_end_of_page(driver)
    driver = click_to_carets(driver)
    driver = write_replies(driver, username)
    return driver
