import asyncio
import os
from playwright.async_api import async_playwright

async def convert_html_to_pdf(html_file_path, output_pdf_path):
    async with async_playwright() as p:
        # Launch the browser
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Get the absolute path for the HTML file
        abs_path = os.path.abspath(html_file_path)
        url = f"file://{abs_path}"

        # Load the HTML file
        await page.goto(url, wait_until="networkidle")

        # Get the total height of the content to make it one long page
        dimensions = await page.evaluate('''() => {
            return {
                width: document.documentElement.offsetWidth,
                height: document.documentElement.scrollHeight
            }
        }''')

        # Convert to PDF with custom height (one long page)
        # We add a small buffer to the height to ensure nothing is clipped
        await page.pdf(
            path=output_pdf_path,
            width=f"{dimensions['width']}px",
            height=f"{dimensions['height'] + 20}px",
            print_background=True,
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"}
        )

        await browser.close()
        print(f"Successfully converted {html_file_path} to {output_pdf_path} (Long Page)")

if __name__ == "__main__":
    input_html = "restaurant.html"
    output_pdf = "restaurant_proposal.pdf"
    
    if os.path.exists(input_html):
        try:
            asyncio.run(convert_html_to_pdf(input_html, output_pdf))
        except Exception as e:
            print(f"Error: {e}")
            print("\nTo run this script, you may need to install playwright:")
            print("pip install playwright")
            print("playwright install chromium")
    else:
        print(f"File not found: {input_html}")
