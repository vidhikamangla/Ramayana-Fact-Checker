# Brief Algorithm On what is being done
#Selenium has been used to control a web browser and Pandas to organize the data. 
#We also use regular expressions to extract verse numbers from the text.

#Opening the Website:
#The script opens the main page of valmikiramayan.net and switches to the frame that contains the main content.

#Finding All Books (Kandas):
#It looks for all the links to the different books of the Ramayana on the main page and saves their names and links.

#Looping through each book:
#For each book:
#It prints which book is being processed.
#It opens the book’s page and finds all the chapters inside that book.

#Looping through each chapter
#For each chapter in the book:
#It opens the chapter’s page.
#It again switches to the correct frame to access the main content.

#Extracting Verses and Translations
#For special books like "Sundara Kanda" and "Yuddha Kanda", it finds verses and their translations using specific HTML tags.
#For other books, it searches for paragraphs containing English translations, and uses regular expressions to extract the verse numbers from the text.

#Storing the Data:
#For every verse and its translation, it saves the following information:
#Book name (Kanda)
#Chapter number (Sarga)
#Verse number (Shloka)
#English translation

#Handling Errors:
#If anything goes wrong while processing a book, the script prints out an error message but continues with the next book.

#Saving everything to excel:
#Once all the data is collected, it saves everything into an Excel file called valmiki_ramayana_dataset.xlsx.

# ----------------------------------------------------------------------------------------------------------------
#Importing necessary libraries
# ----------------------------------------------------------------------------------------------------------------

# We import:
# - `selenium` to automate web browsing,
# - `pandas` to store scraped data in Excel,
# - `re` for regular expression-based text cleaning.
# - logging for error handling and batch processing

from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import re
import logging
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#setting up logging for keeping a track of all the the events when this program runs
logging.basicConfig(filename="bot_log.log",format="%(asctime)s - %(message)s",level=logging.DEBUG)
logger=logging.getLogger()

# ----------------------------------------------------------------------------
# Defined a function to extract verse numbers**
# ----------------------------------------------------------------------------

# Sometimes verse numbers look like `1-2`, `1;2a`, or `3.4`. We use regex to:
# - Capture the number portion at the start,
# - Replace semicolons/hyphens with commas,
# - Remove periods,
# - Clean up surrounding commas.

def extract_verse_number(em_text):
    match = re.match(r'^([\d;,\-\.]+)', em_text)  #matches digits and punctuation at the start
    if match:
        verse = match.group(1)
        verse = verse.replace(';', ',').replace('-', ',').replace('.', '')
        return verse.strip(',')  # remove leading commas
    return ''

# ---------------------------------------------
#setting up chrome web driver  
# ------------------------------------------------
#this is what lets us control the browser to scrape the website.

def setup_driver():
    try:
        chrome_options = webdriver.ChromeOptions()
        prefs = {"download.prompt_for_download": False,
                 "safebrowsing.enabled": True}

        chrome_options.add_experimental_option('prefs', prefs)
        driver = webdriver.Chrome(options=chrome_options)
        logger.info("-------- Set up WebDriver ") 
        return driver
    except Exception as e:
        logger.error(f"Error setting up WebDriver: {e}")
        raise
# ----------------------------------------------------------
#opening main page and collecting all Book (Kanda) links
# ----------------------------------------------------------

# The books are inside a `<frame name="main">`, so we switch to that frame.
# Then we extract all `<a>` tags inside `<li>` elements to get Kanda names and links.

def get_kandas(driver):
    try:
        driver.maximize_window()
        driver.get(f"https://valmikiramayan.net/")
        driver.switch_to.frame("main")
        elements = driver.find_elements(By.CSS_SELECTOR, "li a")  #Book links in <li><a>
        book_links = [{"text": ele.text, "href": ele.get_attribute("href")} for ele in elements]
        logger.info("-------- EXtracted all the book links.") 
        return book_links
    except Exception as e:
        logger.error(f"Error extracting book links: {e}")
        raise

# ----------------------------------------------------------
#Loop through each kanda, visit its chapters, and extract links
# -----------------------------------------------------------

# For each Kanda:
# -we reopen homepage to reset frame state,
# -navigate to the book link,
# -find all chapter links by checking multiple known XPaths.
  
def get_sargas(driver, book):
    try:
        print(f"Processing book: {book['text']}")
        driver.get("https://www.valmikiramayan.net/")
        driver.switch_to.default_content()
        driver.switch_to.frame("main")
        kanda = book["text"]
        driver.get(book["href"])

        #trying known locations for Sarga links
        chapter_links = []
        chapters = driver.find_elements(By.XPATH, "//center[2]/table/tbody/tr/td[2]/a")
        if not chapters:
            chapters = driver.find_elements(By.XPATH, "//center/table/tbody/tr/td/center/table/tbody/tr/td[2]/a")
        if not chapters:
            chapters = driver.find_elements(By.XPATH, "//center/center/table/tbody/tr/td[2]/a")
        for chapter in chapters:
            chapter_links.append({"text": chapter.text, "href": chapter.get_attribute("href")})
        logger.info(f"-------- Extracted all the chapter links for book{book['text']}.") 
        return chapter_links,kanda
    except Exception as e:
        print(f'Error extracting chapter links for book {book['text']}: {e}')

# ----------------------------------------------------------
#Loop through each chapter (Sarga)
# -----------------------------------------------------------

# For each Sarga:
# -we open the chapter
# -check if it's If it’s Sundara or Yuddha Kanda:
    #   - Use `<p class="pratipada"> <em>` for verse numbers,
    #   - Use `<p class="tat">` for translations.
# -otherwise:
    #   - Both verse and translation are in the `<p class="tat">`, with verse shown at the end inside brackets.

def get_verses(ch_num,driver, chap,kanda):
    try:
        sarga = ch_num
        data=[]
        driver.get(chap["href"])
        # element=WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "p.tat"))) 
        driver.switch_to.default_content()
        driver.switch_to.frame("main")

        if kanda.strip().startswith("Sundara Kanda") or kanda.strip().startswith("Yuddha Kanda") or kanda.strip().startswith("Ayodhya Kanda"):
            pratipada_ps = driver.find_elements(By.CSS_SELECTOR, "p.pratipada")  #for verse number
            tat_paragraphs = driver.find_elements(By.CSS_SELECTOR, "p.tat")       #for translation

            for pratipada_p, trans_elem in zip(pratipada_ps, tat_paragraphs):
                ems = pratipada_p.find_elements(By.TAG_NAME, "em")
                verse_numbers = extract_verse_number(ems[0].text.strip()) if ems else ''
                translation = trans_elem.text.strip()

                data.append({
                    "Kanda/Book": kanda,
                    "Sarga/Chapter": sarga,
                    "Shloka/Verse Number": verse_numbers,
                    "English Translation": translation
                })
            return data
        # handling chapters where verse and translation are combined
        # look for for `[1-2a]` or similar patterns at the end of the paragraph.
        #  extract that as verse number and remove it from the text.
        else:
            paragraphs = driver.find_elements(By.CSS_SELECTOR, "p.tat")
            for para in paragraphs:
                text = para.text.strip()
                matches = re.findall(r'\[(\d[\d\-\,\sa-zA-Z\.]*)\]', text)

                if matches:
                    verse_ref = matches[-1].strip().replace('.', ',')
                    parts = verse_ref.split('-')
                    verse_numbers = parts[-1] if len(parts) >= 3 else verse_ref
                    cleaned_text = re.sub(r'\s*\[' + re.escape(matches[-1]) + r'\]\s*$', '', text)

                    data.append({
                        "Kanda/Book": kanda,
                        "Sarga/Chapter": sarga,
                        "Shloka/Verse Number": verse_numbers,
                        "English Translation": cleaned_text
                    })
            return data
    except Exception as e:
        print(f'Error extracting verses for book: {kanda}, chapter: {sarga} : {e}')

# ------------------------------------------------------------------------------------
driver=setup_driver()
book_links=get_kandas(driver)
data = []
for book in book_links:
    ch_num=1
    chapter_links,kanda=get_sargas(driver, book)
    for chapter in chapter_links:
        verses=get_verses(ch_num,driver, chapter, kanda)
        ch_num+=1
        data+=verses
driver.quit()

# ------------------------------------------
#Saving all scraped data to Excel
# ------------------------------------------

#Once we’ve gone through all books and chapters, we save everything in a spreadsheet.
df = pd.DataFrame(data)
df.to_excel("valmiki_ramayana_dataset.xlsx", index=False)
print("data scraping complete.")