from selenium import webdriver
import logging

from selenium.webdriver.common.by import By
import pandas as pd
import re

#setting up logging for keeping a track of all the the events when this program runs
logging.basicConfig(filename="bot_log.log",format="%(asctime)s - %(message)s",level=logging.DEBUG)
logger=logging.getLogger()

#setting up driver and opening chrome 
def setup_driver():
    try:
        chrome_options = webdriver.ChromeOptions()
        prefs = {"download.prompt_for_download": False,
                 "safebrowsing.enabled": True}

        chrome_options.add_experimental_option('prefs', prefs)
        driver = webdriver.Chrome(options=chrome_options)
        driver.maximize_window()
        driver.get(f"https://valmikiramayan.net/")
        driver.switch_to.frame("main")
        logger.info("-------- Set up WebDriver ")
        book_links=kanda_extract(driver)
        data=verse_extract(driver,book_links)
        driver.quit()  
        return driver,data
    except Exception as e:
        logger.error(f"Error setting up WebDriver: {e}")
        raise
    
def extract_verse_number(em_text):
    match = re.match(r'^([\d;,\-\.]+)', em_text)
    if match:
        verse = match.group(1)
        verse = verse.replace(';', ',')  
        verse = verse.replace('-', ',')  
        verse = verse.replace('.', '')   
        verse = verse.strip(',')
        return verse
    return ''

def kanda_extract(driver):
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, "li a")
        book_links = [{"text": ele.text, "href": ele.get_attribute("href")} for ele in elements]
        logger.info("-------------Extracted all the kanda names and book links")
        return book_links
    
    except Exception as e:
        logger.error(f"Error extracting kanda names: {e}")

def verse_extract(driver,book_links):
    data = []
    for book in book_links:
        try:
            print(f"Processing book: {book['text']}")

            driver.get("https://www.valmikiramayan.net/")
            driver.switch_to.default_content()
            driver.switch_to.frame("main")

            kanda = book["text"]

            driver.get(book["href"])

            chapter_links = []
            chapters = driver.find_elements(By.XPATH, "//center[2]/table/tbody/tr/td[2]/a")
            if len(chapters) == 0:
                chapters = driver.find_elements(
                    By.XPATH, "//center/table/tbody/tr/td/center/table/tbody/tr/td[2]/a"
                )
                if len(chapters) == 0:
                    chapters = driver.find_elements(
                        By.XPATH, "//center/center/table/tbody/tr/td[2]/a"
                    )

            for chapter in chapters:
                chapter_links.append(
                    {"text": chapter.text, "href": chapter.get_attribute("href")}
                )

            count = 1

            for chap in chapter_links:
                sarga = count
                driver.get(chap["href"])
                driver.switch_to.default_content()
                driver.switch_to.frame("main")
                count += 1

                if kanda.strip().startswith("Sundara Kanda") or kanda.strip().startswith("Yuddha Kanda"):
                    pratipada_ps = driver.find_elements(By.CSS_SELECTOR, "p.pratipada")
                    tat_paragraphs = driver.find_elements(By.CSS_SELECTOR, "p.tat")

                    for pratipada_p, trans_elem in zip(pratipada_ps, tat_paragraphs):
                        ems = pratipada_p.find_elements(By.TAG_NAME, "em")
                        if ems:
                            em_text = ems[0].text.strip()
                            verse_numbers = extract_verse_number(em_text)
                        else:
                            verse_numbers = ''
                        translation = trans_elem.text.strip()
                        data.append({
                            "Kanda/Book": kanda,
                            "Sarga/Chapter": sarga,
                            "Shloka/Verse Number": verse_numbers,
                            "English Translation": translation
                        })

                else:
                    paragraphs = driver.find_elements(By.CSS_SELECTOR, "p.tat")
                    for para in paragraphs:
                        text = para.text.strip()
                        matches = re.findall(r'\[(\d[\d\-\,\sa-zA-Z\.]*)\]', text)
                        if matches:
                            verse_ref = matches[-1].strip().replace('.', ',')  
                            parts = verse_ref.split('-')
                            if len(parts) >= 3:
                                verse_numbers = parts[-1]
                            else:
                                verse_numbers = verse_ref
                            cleaned_text = re.sub(r'\s*\[' + re.escape(matches[-1]) + r'\]\s*$', '', text)
                            data.append({
                                "Kanda/Book": kanda,
                                "Sarga/Chapter": sarga,
                                "Shloka/Verse Number": verse_numbers,
                                "English Translation": cleaned_text
                            })
            logger.info("----------extraction complete")
        except Exception as e:
            logger.error(f"Error extracting the verses.: {e}")
            
    return data

driver, data=setup_driver()
df = pd.DataFrame(data)
df.to_excel("valmiki_dataset.xlsx", index=False)

print("data scraping complete. saved as excel file")
logger.info("----------extraction complete")

driver.quit()