import requests
from bs4 import BeautifulSoup
import PyPDF2
import time


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def load_text_from_url(url: str) -> list[str]:
    """Fetches content from a URL and returns it as text chunks."""
    try:
        print(f"🔗 Fetching content from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        text_elements = soup.find_all(["p", "h1", "h2", "h3", "li"])
        full_text = "\n".join([elem.get_text() for elem in text_elements])

        chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
        print(f"  ✅ Found {len(chunks)} text chunks.")
        return chunks
    except requests.RequestException as e:
        print(f"  ❌ Error fetching URL {url}: {e}")
        return []


def load_text_from_pdf(file_path: str) -> list[str]:
    """Extracts text from a PDF file and returns it as text chunks."""

    try:
        print(f"📄 Reading content from {file_path}...")
        full_text = ""
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                full_text += page.extract_text() + "\n\n"

        chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
        print(f"  ✅ Found {len(chunks)} text chunks.")
        return chunks
    except Exception as e:
        print(f"  ❌ Error reading PDF {file_path}: {e}")
        return []


def get_development_links(hub_url: str) -> list[str]:
    """
    Scrapes a hub page using Selenium with intelligent waits to find all links.
    """
    print(f"🔎 Scraping for development links at {hub_url} using Selenium...")

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--log-level=3")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    links = set()
    try:
        driver.get(hub_url)

        try:

            cookie_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            cookie_button.click()
            print("  👍 Clicked accept on the cookie banner.")

            time.sleep(2)
        except Exception:
            print("  ℹ️ No cookie banner found, or could not click it. Continuing...")

        wait = WebDriverWait(driver, 15)
        wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "a.project-card-listing__card")
            )
        )

        soup = BeautifulSoup(driver.page_source, "html.parser")

        for a_tag in soup.select("a.project-card-listing__card"):
            href = a_tag.get("href")
            if href:
                links.add(href)

        print(f"  ✅ Found {len(links)} unique development links.")

    except Exception as e:
        print(f"  ❌ Error scraping hub URL with Selenium {hub_url}: {e}")
    finally:
        driver.quit()

    return list(links)
