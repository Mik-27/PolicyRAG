import requests
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from utils.utils import checkUrlHealth

class WebScraper:
    def __init__(self, url):
        self.url = url


    # def checkURL(self):
    #     """
    #         Check whether provided URL is valid or not.
    #     """
    #     try:
    #         response = requests.get(self.url)
    #         if response.status_code == 200:
    #             return 1
    #         else:
    #             response.raise_for_status()
    #             return 0
    #     except requests.exceptions.RequestException as e:
    #         raise SystemExit(e)
        
        
    def getFolderNames(self, driver):
        """
            Get all the folder names to parse.
        """
        try:
            div = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.ReactVirtualized__Grid__innerScrollContainer'))
            )
            head = div.find_element(By.ID, 'Folder|null')
            folder_names = list(head.get_attribute("aria-owns").split(" "))
            return folder_names
            
        except:
            raise Exception("Error getting folder names.")
        
    def getFileNames(self, driver, folder):
        """
            Get all the file names from the sub-directory
        """
        try:
            div = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.ReactVirtualized__Grid__innerScrollContainer'))
            )
            head = div.find_element(By.ID, folder)
            file_names = list(head.get_attribute("aria-owns").split(" "))
            return file_names
            
        except:
            raise Exception("Error getting file names.")
          
        
    def downloadPdfFromIframe(self, driver, element, file):
        """
            Download a PDF from Iframe element
        """
        doc = driver.find_element(By.ID, file)
        doc.click()
        time.sleep(5)

        # Switch to iframe
        driver.switch_to.frame(driver.find_element(By.ID, "webviewer-1"))
        
        menu_ = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//button[@data-element='menuButton']"))
        )
        menuButton = driver.find_element(By.XPATH, "//button[@data-element='menuButton']")
        menuButton.click()
        time.sleep(3)
        menuDropdown = driver.find_element(By.XPATH, "//div[@data-element='menuOverlay']")
        downloadButton = menuDropdown.find_element("xpath", "//button[@data-element='downloadButton']")
        downloadButton.click()
        driver.switch_to.default_content()
        file_name = file.split("|")[1]
        print("Downloaded - " + file_name + ".pdf")
        

    def getDocuemnts(self):
        try:
            if checkUrlHealth(self.url):
                service = Service(executable_path='./chromedriver/chromedriver.exe')
                options = webdriver.ChromeOptions()
                prefs = {"profile.default_content_settings.popups": 0,
                    "download.default_directory": r"C:\Mihir\Projects\RAG\documents\\",
                    "directory_upgrade": True}
                options.add_experimental_option("prefs",prefs)
                driver = webdriver.Chrome(service=service, options=options)
                driver.get(self.url)

                _ = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'span.defaultDmsObjectTreeItem_collapseIcon'))
                )
                drop1 = driver.find_element(By.CSS_SELECTOR, 'span.defaultDmsObjectTreeItem_collapseIcon')
                drop1arrow = drop1.find_element(By.TAG_NAME, 'span')
                master_div = driver.find_element(By.CSS_SELECTOR, 'div.ReactVirtualized__Grid__innerScrollContainer')

                if drop1arrow.get_attribute('data-automation-id') == "dmsIcon-TreeItemExpanded":
                    folder_names = self.getFolderNames(driver)
                    for folder_name in folder_names:
                        folder = master_div.find_element(By.ID, folder_name)
                        expand_button = folder.find_element(By.TAG_NAME, "button")
                        expand_button.click()
                        file_names = self.getFileNames(driver, folder_name)

                        time.sleep(2)

                        for file in file_names[:2]:
                            self.downloadPdfFromIframe(driver, element=master_div, file=file)
                            time.sleep(1)

                        # Close subdirectory dropdown
                        expand_button.click()
                        time.sleep(2)
                    
                else:
                    # Drop root folder
                    drop1arrow.click()
                    time.sleep(2)



            else:
                raise Exception("Invalid URL.")
            
        except:
            raise Exception("Error pasrsing URL.")
        finally:
            driver.close()


if __name__ == "__main__":
    url = "https://public.powerdms.com/ASU/tree"
    print("Scraping data from " + url)
    ws = WebScraper(url)
    ws.getDocuemnts()