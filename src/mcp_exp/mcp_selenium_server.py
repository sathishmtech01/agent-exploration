# mcp_selenium_server.py
from mcp.server.fastmcp import FastMCP
from selenium import webdriver
from selenium.webdriver.common.by import By

mcp = FastMCP("Selenium Automation Server")
driver = None

@mcp.tool()
def start_browser(browser_name: str = "chrome"):
    """Starts the specified browser (chrome or firefox)."""
    global driver
    if browser_name.lower() == "chrome":
        driver = webdriver.Chrome()
    elif browser_name.lower() == "firefox":
        driver = webdriver.Firefox()
    return f"{browser_name} browser started."

@mcp.tool()
def navigate_to(url: str):
    """Navigates to the specified URL."""
    if driver:
        driver.get(url)
        return f"Navigated to {url}"
    return "Browser not started."

@mcp.tool()
def quit_browser():
    """Quits the current browser session."""
    global driver
    if driver:
        driver.quit()
        driver = None
        return "Browser closed."
    return "No browser open."

# You can add more tools for clicking, typing, etc.

if __name__ == "__main__":
    mcp.run()
