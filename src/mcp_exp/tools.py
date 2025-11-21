# generic_web_agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = None

def start_browser_tool(browser_name: str = "chrome") -> str:
    """Starts the specified browser (chrome or firefox)."""
    global driver
    if browser_name.lower() == "chrome":
        driver = webdriver.Chrome()
    elif browser_name.lower() == "firefox":
        driver = webdriver.Firefox()
    return f"{browser_name} browser started."

def navigate_to_tool(url: str) -> str:
    """Navigates the browser to a specific URL."""
    if driver:
        driver.get(url)
        return f"Navigated to {url}"
    return "Browser not running."

def click_element_tool(xpath_selector: str) -> str:
    """Finds an element by its XPath selector and clicks it."""
    if driver:
        # Wait for the element to be clickable
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, xpath_selector))
        )
        element.click()
        return f"Clicked element with XPath: {xpath_selector}"
    return "Browser not running."

def type_into_tool(xpath_selector: str, text: str) -> str:
    """Finds a text input element by its XPath selector and types text into it."""
    if driver:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, xpath_selector))
        )
        element.clear() # Clear existing text
        element.send_keys(text)
        return f"Typed into element with XPath: {xpath_selector}"
    return "Browser not running."

def extract_text_tool(xpath_selector: str) -> str:
    """Extracts visible text content from elements matching an XPath selector."""
    if driver:
        elements = driver.find_elements(By.XPATH, xpath_selector)
        text_content = [el.text for el in elements if el.text.strip()]
        if text_content:
            return "Extracted Text:\n" + "\n---\n".join(text_content)
        return "No text found for the given XPath."
    return "Browser not running."

def quit_browser_tool() -> str:
    """Quits the current browser session."""
    global driver
    if driver:
        driver.quit()
        driver = None
        return "Browser closed."
    return "No browser open."

# Create ADK FunctionTools
start_browser_adk = FunctionTool(start_browser_tool)
navigate_adk = FunctionTool(navigate_to_tool)
click_adk = FunctionTool(click_element_tool)
type_adk = FunctionTool(type_into_tool)
extract_adk = FunctionTool(extract_text_tool)
quit_browser_adk = FunctionTool(quit_browser_tool)
