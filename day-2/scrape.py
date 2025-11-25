# =============================================================================
# Generic Web Scraper - Foundation Script
# =============================================================================
# This script provides a flexible, reusable web scraping solution that works
# with any website. It intelligently identifies content areas using multiple
# fallback strategies and formats the extracted text in a clean, readable way.
#
# Key Features:
# - Automatic content detection using multiple CSS selectors
# - Smart filtering to remove navigation, headers, and footers
# - Duplicate content detection and removal
# - Markdown-style formatting for headers and lists
# - Support for both single and batch scraping
# =============================================================================

import requests  # HTTP library for fetching web pages
from bs4 import BeautifulSoup  # HTML parsing library
import re  # Regular expressions for text cleaning


def scrape_website(url):
    """
    Generic website scraper that works with any URL
    
    This is the core scraping function that implements an intelligent content
    detection strategy. It tries multiple CSS selectors in order of specificity,
    starting with framework-specific selectors (like Elementor) and falling back
    to generic ones (like 'body'). This makes it adaptable to various website
    structures without requiring custom configuration.

    Args:
        url: The URL to scrape

    Returns:
        Extracted text content formatted as markdown, or None if scraping failed
    
    Content Extraction Strategy:
    1. Send HTTP request with browser-like headers to avoid blocking
    2. Parse HTML with BeautifulSoup
    3. Try multiple CSS selectors to find main content area
    4. Extract and format text while filtering out navigation/footer elements
    5. Remove duplicates and clean up whitespace
    6. Return formatted markdown-style text
    """
    # Set browser-like headers to avoid being blocked by websites
    # Many sites reject requests from scripts, but accept browser requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Send GET request to fetch the webpage
        response = requests.get(url, headers=headers)
        # Raise an exception for HTTP error codes (4xx, 5xx)
        response.raise_for_status()

        # Parse the HTML content using BeautifulSoup
        # 'html.parser' is Python's built-in parser (no external dependencies)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Define a prioritized list of CSS selectors to locate main content
        # Ordered from most specific (framework-specific) to most general (body)
        # This cascading approach ensures we find content on diverse site structures
        content_selectors = [
            'div[data-elementor-type="single-post"]',  # Elementor page builder (WordPress)
            '.entry-content',  # Standard WordPress content wrapper
            '.post-content',   # Alternative WordPress content class
            'main article',    # Semantic HTML5: article within main
            'main',            # HTML5 main content element
            'article',         # HTML5 article element
            '.content',        # Generic content class (common convention)
            '.main-content',   # Alternative generic content class
            '#content',        # Content by ID (common convention)
            'body'             # Ultimate fallback - entire page body
        ]

        # Try each selector until we find content
        # select_one() returns the first matching element or None
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break  # Found content, stop searching

        # Safety check: if all selectors failed, try to find body element
        if not main_content:
            main_content = soup.find('body')

        # If we still have no content, the page structure is unexpected
        if not main_content:
            return None

        # Extract the page title from the <title> tag
        # This gives context about what the page is about
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""

        # Extract the main heading (h1 or h2) to compare with title
        # Often the title and h1 are the same, so we'll deduplicate later
        main_heading = soup.find(['h1', 'h2'])
        heading_text = main_heading.get_text().strip() if main_heading else ""

        # Find all elements that typically contain meaningful content
        # We include headers (h1-h6), paragraphs (p), list items (li),
        # and containers (div, span) that might have text
        content_elements = main_content.find_all([
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div', 'span'
        ])

        # Initialize containers for processed content
        text_parts = []  # List to store formatted text segments
        processed_texts = set()  # Set to track and prevent duplicate content

        # Add the page title as a top-level heading, but only if it's different
        # from the main heading to avoid redundancy
        if title_text and title_text != heading_text:
            text_parts.append(f"# {title_text}\n")

        # Process each content element, applying filters and formatting
        for element in content_elements:
            # Skip elements that are inside navigation, footer, or header sections
            # These typically contain site-wide UI elements, not content
            if element.find_parent(['nav', 'footer', 'header']):
                continue

            # Also skip elements with class names that indicate non-content areas
            # Join all classes into a single lowercase string for easy checking
            classes = ' '.join(element.get('class', [])).lower()
            if any(skip in classes for skip in ['nav', 'menu', 'footer', 'sidebar', 'widget']):
                continue

            # Extract and clean the text from the element
            text = element.get_text().strip()
            # Skip empty elements or those with very short text (likely UI fragments)
            if not text or len(text) < 10:
                continue

            # Prevent duplicate content from appearing multiple times
            # Normalize whitespace (collapse multiple spaces) before checking
            normalized = ' '.join(text.split())
            if normalized in processed_texts:
                continue  # Already processed this text, skip it
            processed_texts.add(normalized)  # Mark as processed

            # Format the text based on its HTML element type
            # This preserves document structure in the output
            if element.name.startswith('h'):
                # Heading elements: convert to markdown heading syntax
                # Extract the heading level (1-6) from element name (h1, h2, etc.)
                level = int(element.name[1]) if element.name[1].isdigit() else 2
                # Create markdown heading prefix (e.g., ## for h2)
                prefix = '#' * min(level, 6)  # Cap at 6 levels
                text_parts.append(f"\n{prefix} {text}\n")
            elif element.name == 'li':
                # List items: format as markdown bullet points
                text_parts.append(f"- {text}")
            else:
                # Regular paragraph or div: add as plain text
                text_parts.append(text)

        # Combine all text parts and perform final cleanup
        final_text = '\n'.join(text_parts)
        # Remove excessive blank lines (replace 3+ newlines with just 2)
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)
        # Collapse multiple spaces into single spaces
        final_text = re.sub(r' {2,}', ' ', final_text)

        # Return the cleaned, formatted text
        return final_text.strip()

    except Exception as e:
        # Catch any errors (network issues, parsing errors, etc.)
        print(f"Error scraping {url}: {e}")
        return None


def scrape_multiple_websites(urls):
    """
    Batch scraping function for processing multiple websites
    
    This convenience function wraps scrape_website() to handle multiple URLs
    in a single call. It provides flexible input handling (list or dict) and
    returns a labeled dictionary of results.

    Args:
        urls: Either a list of URLs ['url1', 'url2', ...]
              Or a dict with labels {'Label 1': 'url1', 'Label 2': 'url2', ...}

    Returns:
        Dict mapping labels to scraped content:
        {'Label 1': 'content...', 'Label 2': 'content...', ...}
    """
    results = {}

    # Flexible input handling: convert list to dict with auto-generated labels
    if isinstance(urls, list):
        # Create labels like "Site 1", "Site 2", etc.
        url_dict = {f"Site {i+1}": url for i, url in enumerate(urls)}
    else:
        # Already a dict, use as-is
        url_dict = urls

    # Iterate through each URL and scrape it
    for label, url in url_dict.items():
        print(f"\nScraping {label}...")

        # Use the core scrape_website function
        content = scrape_website(url)
        if content:
            results[label] = content
            print(f"✓ Successfully scraped {label} ({len(content)} characters)")
        else:
            print(f"✗ Failed to scrape {label}")

    return results


# =============================================================================
# Demo Section - Shows both single and batch scraping
# =============================================================================
if __name__ == "__main__":
    # Demo 1: Single website scraping
    # Shows how to scrape a single URL and preview the results
    print("=== SINGLE WEBSITE DEMO ===")
    url = "https://gdpr-info.eu/art-17-gdpr/"
    content = scrape_website(url)

    if content:
        print("\n" + "=" * 50)
        print("Preview of scraped content:")
        print("=" * 50)
        # Show first 500 characters as a preview
        print(content[:500] + "..." if len(content) > 500 else content)
        print(f"\nTotal characters scraped: {len(content)}")

    # Demo 2: Batch scraping multiple websites
    # Shows how to scrape several URLs at once using a list
    print("\n\n=== MULTIPLE WEBSITES DEMO ===")
    # List of GDPR articles to scrape (Articles 17, 18, and 19)
    websites_to_scrape = [
        "https://gdpr-info.eu/art-17-gdpr/",
        "https://gdpr-info.eu/art-18-gdpr/",
        "https://gdpr-info.eu/art-19-gdpr/"
    ]

    # Scrape all websites and collect results
    multiple_content = scrape_multiple_websites(websites_to_scrape)

    # Display detailed summaries of each scraped site
    if multiple_content:
        print("\n" + "=" * 60)
        print("DETAILED SUMMARIES")
        print("=" * 60)
        # Iterate through results and show summary for each
        for label, content in multiple_content.items():
            print(f"\n{'-' * 40}")
            print(f"📄 {label}")
            print(f"{'-' * 40}")
            if content:
                # Extract title from the content (first substantial line)
                lines = content.split('\n')
                # Find the first line that's not empty and longer than 5 chars
                title = next((line.strip('# ') for line in lines if line.strip() and len(line.strip()) > 5), "No title")

                print(f"Title: {title}")
                print(f"Length: {len(content)} characters")
                print(f"Preview:")
                # Show first 400 characters as preview
                print(content[:400] + "..." if len(content) > 400 else content)
            else:
                print("❌ No content available")
    else:
        print("No websites were successfully scraped.")