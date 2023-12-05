import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By

import urllib
import time
import os

# Configure WebDriver
driver_path = "C:\\chromedriver\\chromedriver.exe"  # Replace with the actual path to the downloaded driver
car_list = [
     "toyota+RAV4",
     "toyota+86",
     "toyota+Sienna",
    # "toyota+Camry",
    # "toyota+C-HR",
    # "toyota+Corolla+sedan",
    # "toyota+4Runner",
    # "toyota+Venza"
]

views = {
    'stock+photos' : 90,
    "front+view" : 90,
    "side+profile" : 90,
    "back+angle+view" : 90,
    "on+the+road" : 90,
    "tailight+view+photoshoot" : 20,
    "headlight+view+photoshoot" : 20,
    "modifications+photoshoot" : 90
    }
    

# Launch Browser and Open the URL
counter = 0
for car_model in car_list:
    for angle, numOfPics in views.items():
        options = webdriver.ChromeOptions()
        options.add_argument('--user-data-dir=C:\\Users\\novojit\\AppData\\Local\\Google\\Chrome\\User Data')
        #options.add_argument('--headless')
        driver = uc.Chrome(options=options)
        driver.minimize_window()

        # Create url variable containing the webpage for a Google image search.
        # url = "https://www.google.com/search?q={s}&tbm=isch&tbs=sur%3Afc&hl=en&ved=0CAIQpwVqFwoTCKCa1c6s4-oCFQAAAAAdAAAAABAC&biw=1251&bih=568"
        #url = str("https://www.google.com/search?q={0}+{1}&hl=en&tbm=isch&sxsrf=APwXEdeMCCcn15mo1obWv-xVcr_tpnFYQg%3A1684476865544&source=hp&biw=1737&bih=1032&ei=wRNnZNX-HtX4kPIP9umT2AY&iflsig=AOEireoAAAAAZGch0VXQnHgSIAIKBwcg5h0gf-nJjQvD&oq=toyota+supr&gs_lcp=CgNpbWcQAxgAMgQIIxAnMgQIIxAnMggIABCABBCxAzIICAAQgAQQsQMyBQgAEIAEMggIABCABBCxAzIICAAQgAQQsQMyCAgAEIAEELEDMggIABCABBCxAzIICAAQgAQQsQM6BwgjEOoCECc6CAgAELEDEIMBOgQIABADOgkIABAYEIAEEApQlglY7SNgpSxoB3AAeAGAAZIBiAHkDJIBBDEwLjeYAQCgAQGqAQtnd3Mtd2l6LWltZ7ABCg&sclient=img".format(car_model, angle))
        url = str("https://www.google.com/search?q={0}+{1}&tbm=isch&ved=2ahUKEwjFher07PeCAxWVa2wGHWaFAoUQ2-cCegQIABAA&oq=toyota+front+view&gs_lcp=CgNpbWcQAzIFCAAQgAQyBQgAEIAEMgYIABAIEB4yBggAEAgQHjIGCAAQCBAeMgYIABAIEB4yBggAEAgQHjIGCAAQCBAeMgYIABAIEB4yBggAEAgQHjoKCAAQgAQQigUQQzoICAAQgAQQsQNQnQVY7X5g3YABaAJwAHgAgAGdAYgBkhSSAQQwLjE4mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=nttuZcW-C5XXseMP5oqKqAg&bih=690&biw=1042".format(car_model, angle))
        # Launch the browser and open the given url in your webdriver.
        # search_query = "Toyota Supra"

        #[general], front view, rear view, side profile, back angle view, on the road, tail-lights, headlights
        driver.get(url)


        # The execute script function will scroll down the body of the web page and load the images.
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
        time.sleep(4)
        if (numOfPics > 50):
            for i in range(0, 3):
                driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
                time.sleep(2)
        elif (numOfPics > 20):
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            time.sleep(2)

        # Review the Web Page’s HTML Structure

        # We need to understand the structure and contents of the HTML tags and find an attribute that is unique only to images.
        img_results = driver.find_elements(By.XPATH, "//img[contains(@class, 'Q4LuWd')]")

        image_urls = []
        for img in img_results:
            image_urls.append(img.get_attribute('src'))

        folder_path = 'C:\\ScrapeDataset' # change your destination path here

        modifiedName = car_model.replace("+", "_")

        for i in range(min(numOfPics, len(image_urls))):
            counter += 1
            if image_urls[i] is not None:
                file_path = os.path.join(folder_path, "{0} {1}.jpg".format(modifiedName, counter))
                urllib.request.urlretrieve(str(image_urls[i]), file_path)

        driver.quit()
    counter = 0