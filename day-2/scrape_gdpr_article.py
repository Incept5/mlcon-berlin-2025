# =============================================================================
# GDPR-Specific Web Scraper - Specialized Extension
# =============================================================================
# This script is a SPECIALIZED VERSION of the generic scraper (scrape.py).
# While scrape.py provides general-purpose scraping for any website, this
# script is tailored specifically for scraping GDPR articles from gdpr-info.eu.
#
# RELATIONSHIP TO scrape.py:
# - EVOLUTION: This was likely developed BEFORE the generic scraper, serving
#   as the initial prototype that informed the more flexible design
# - SPECIALIZATION: Includes GDPR-specific features like article heading
#   detection and table handling
# - FILE OUTPUT: Saves results to disk (gdpr_article_content.txt) whereas
#   the generic version returns content in memory
# - MORE VERBOSE: Includes additional debug output and error handling
#
# KEY DIFFERENCES FROM scrape.py:
# 1. Saves output to file instead of just returning it
# 2. More detailed console output during scraping
# 3. Special handling for GDPR article headings (e.g., "Article 17")
# 4. Enhanced table extraction with proper formatting
# 5. More granular error messages for debugging
#
# WHEN TO USE THIS vs scrape.py:
# - Use THIS: When scraping GDPR articles and you want file output
# - Use scrape.py: For general web scraping or in-memory processing
# =============================================================================

import requests  # HTTP library for fetching web pages
from bs4 import BeautifulSoup  # HTML parsing library
import re  # Regular expressions for text cleaning and pattern matching


def scrape_gdpr_article(url):
    """
    Scrape GDPR article content from gdpr-info.eu with specialized handling
    
    This function is similar to scrape_website() in scrape.py but includes:
    - GDPR-specific article heading detection (regex pattern matching)
    - Enhanced table extraction and formatting
    - Automatic file saving to gdpr_article_content.txt
    - More verbose debug output showing which selectors worked
    
    Args:
        url: URL of the GDPR article to scrape (e.g., https://gdpr-info.eu/art-17-gdpr/)
    
    Returns:
        Extracted and formatted text content, or None if scraping failed
        
    Side Effects:
        Saves the scraped content to 'gdpr_article_content.txt' in the current directory
    """

    # Set browser-like headers to avoid being blocked by websites
    # Same approach as scrape.py - websites often block requests from scripts
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Send HTTP GET request to fetch the webpage
        response = requests.get(url, headers=headers)
        # Raise exception for HTTP errors (4xx, 5xx status codes)
        response.raise_for_status()

        # Parse the HTML content into a navigable tree structure
        # Uses Python's built-in html.parser (same as scrape.py)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Define prioritized list of CSS selectors for finding main content
        # SAME STRATEGY as scrape.py but with fewer selectors (simplified)
        # The order matters: try specific selectors first, fall back to generic ones
        content_selectors = [
            'div[data-elementor-type="single-post"]',  # Elementor (WordPress page builder)
            '.entry-content',  # Standard WordPress content wrapper
            '.post-content',  # Alternative WordPress content class
            'main',  # HTML5 semantic main element
            'article',  # HTML5 semantic article element
            '.content',  # Generic content class
            'body'  # Ultimate fallback - entire page body
        ]

        # Try each selector in order until we find content
        # DIFFERENCE FROM scrape.py: Prints which selector worked (debug output)
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                # This debug output helps understand the site structure
                print(f"Found content using selector: {selector}")
                break  # Found content, stop searching

        # Safety check: if no selector worked, fall back to body
        # DIFFERENCE FROM scrape.py: More explicit console output about fallback
        if not main_content:
            print("Could not find main content container, using entire body")
            main_content = soup.find('body')

        # Process the content if we found it
        if main_content:
            # Initialize list to collect formatted text segments
            text_content = []

            # Extract the page title from <title> tag
            # SAME as scrape.py but always adds it (doesn't check for duplicates)
            title = soup.find('title')
            if title:
                title_text = title.get_text().strip()
                # Format as markdown level 1 heading
                text_content.append(f"# {title_text}\n")

            # GDPR-SPECIFIC FEATURE: Look for article headings with specific pattern
            # This uses regex to find headings like "Article 17", "Article 23", etc.
            # NOT present in generic scrape.py - this is domain-specific logic
            article_heading = soup.find(['h1', 'h2'], string=re.compile(r'Article \d+', re.IGNORECASE))
            if article_heading:
                # Format as markdown level 2 heading
                text_content.append(f"## {article_heading.get_text().strip()}\n")

            # Find all elements that contain meaningful content
            # DIFFERENCE FROM scrape.py: Also includes table cells (td, th)
            # This ensures table data is extracted in the element loop
            content_elements = main_content.find_all([
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',  # All heading levels
                'p', 'li',  # Paragraphs and list items
                'div', 'span',  # Generic containers
                'td', 'th'  # Table cells (data and headers)
            ])

            # Filter and process content elements
            # DIFFERENCE FROM scrape.py: Uses a two-pass approach (filter then process)
            # scrape.py processes elements in a single loop
            filtered_elements = []
            for element in content_elements:
                # Skip elements inside navigation, footer, or header sections
                # These contain site-wide UI elements, not article content
                if element.find_parent(['nav', 'footer', 'header']):
                    continue

                # Also skip elements with class names indicating non-content areas
                # DIFFERENCE FROM scrape.py: Includes 'meta' in skip list (metadata)
                element_classes = element.get('class', [])
                skip_classes = ['nav', 'menu', 'footer', 'sidebar', 'widget', 'meta']
                if any(skip_class in ' '.join(element_classes).lower() for skip_class in skip_classes):
                    continue

                # Extract and validate text content
                text = element.get_text().strip()
                # Skip empty or very short elements (threshold: 3 chars vs 10 in scrape.py)
                if not text or len(text) < 3:
                    continue

                # Extra filter: skip single-character content (likely UI fragments)
                if len(text) == 1:
                    continue

                # Store element and its text for second-pass processing
                filtered_elements.append((element, text))

            # Second pass: format and add content, avoiding duplicates
            # SAME duplicate prevention strategy as scrape.py
            processed_texts = set()  # Track text we've already added
            for element, text in filtered_elements:

                # Prevent duplicate content from appearing multiple times
                if text in processed_texts:
                    continue  # Already added this text
                processed_texts.add(text)  # Mark as processed

                # Format text based on HTML element type
                # SAME markdown formatting strategy as scrape.py
                if element.name and element.name.startswith('h'):
                    # Heading elements: convert to markdown heading syntax
                    level = int(element.name[1]) if element.name[1].isdigit() else 2
                    heading_prefix = '#' * min(level, 6)  # Cap at 6 levels (markdown limit)
                    text_content.append(f"\n{heading_prefix} {text}\n")
                elif element.name == 'li':
                    # List items: format as markdown bullet points
                    text_content.append(f"- {text}")
                elif element.name in ['td', 'th']:
                    # Table cells: skip here, we handle tables separately below
                    # This prevents duplicate table content
                    if element.parent and element.parent.name == 'tr':
                        continue  # Will be processed in table handling section
                else:
                    # Regular paragraph or div: add as plain text
                    text_content.append(text)

            # ENHANCED TABLE HANDLING: Extract tables with proper structure
            # DIFFERENCE FROM scrape.py: This dedicated table extraction is MORE SOPHISTICATED
            # scrape.py doesn't have special table handling
            tables = main_content.find_all('table')
            for table in tables:
                # Add a heading for each table
                text_content.append("\n### Table\n")
                # Process each row in the table
                for row in table.find_all('tr'):
                    # Find all cells (both data cells and header cells)
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        # Join cells with pipe separators for markdown-style tables
                        row_text = ' | '.join(cell.get_text().strip() for cell in cells)
                        if row_text.strip():
                            # Format as markdown table row
                            text_content.append(f"| {row_text} |")
                # Add blank line after table
                text_content.append("")

            # Combine all text segments into final output
            final_text = '\n'.join(text_content)

            # Apply text cleanup using regular expressions
            # DIFFERENCE FROM scrape.py: Allows 3 newlines instead of 2, and removes
            # leading whitespace from each line (more aggressive cleanup)
            final_text = re.sub(r'\n{4,}', '\n\n\n', final_text)  # Max 3 newlines
            final_text = re.sub(r' {2,}', ' ', final_text)  # Collapse multiple spaces
            final_text = re.sub(r'^\s+', '', final_text, flags=re.MULTILINE)  # Remove line-leading whitespace
            final_text = final_text.strip()  # Remove leading/trailing whitespace

            # MAJOR DIFFERENCE FROM scrape.py: Automatically save to file
            # scrape.py returns content in-memory; this writes to disk
            filename = "gdpr_article_content.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(final_text)

            # Provide success feedback with file location
            print(f"Successfully scraped content and saved to {filename}")
            return final_text

        else:
            # No content found (should be rare due to fallbacks)
            print("Could not find any content on the page.")
            return None

    # Error handling with specific exception types
    # DIFFERENCE FROM scrape.py: Separates network errors from other errors
    except requests.exceptions.RequestException as e:
        # Network-related errors (connection issues, timeouts, etc.)
        print(f"Error occurred during scraping: {e}")
        return None
    except Exception as e:
        # Any other unexpected errors (parsing, file I/O, etc.)
        print(f"Unexpected error: {e}")
        return None


def scrape_multiple_gdpr_articles(base_url_pattern, article_numbers):
    """
    Batch scraping function specifically for GDPR articles
    
    COMPARISON TO scrape.py's scrape_multiple_websites():
    - DIFFERENT INTERFACE: Takes URL pattern + article numbers instead of full URLs
    - MORE SPECIALIZED: Designed for numbered GDPR articles (Art. 1, Art. 2, etc.)
    - SAME CONCEPT: Both provide batch scraping with labeled results
    
    This is more convenient for GDPR use cases where articles follow a
    predictable URL pattern: https://gdpr-info.eu/art-{number}-gdpr/

    Args:
        base_url_pattern: URL template with {} placeholder (e.g., "https://gdpr-info.eu/art-{}-gdpr/")
        article_numbers: List of article numbers to scrape (e.g., [17, 18, 19])

    Returns:
        Dict mapping article numbers to scraped content:
        {17: 'content...', 18: 'content...', 19: 'content...', ...}
    """
    all_content = {}

    # Iterate through article numbers and construct URLs
    for article_num in article_numbers:
        # Build the full URL by inserting article number into pattern
        url = base_url_pattern.format(article_num)
        print(f"\nScraping Article {article_num} from {url}")

        # Scrape this article using the specialized function
        content = scrape_gdpr_article(url)
        if content:
            # Store with article number as key for easy reference
            all_content[article_num] = content
            print(f"Successfully scraped Article {article_num}")
        else:
            print(f"Failed to scrape Article {article_num}")

    return all_content


# =============================================================================
# Demo Section
# =============================================================================
if __name__ == "__main__":
    # Demo 1: Single article scraping with file output
    # This demonstrates the basic usage of the specialized GDPR scraper
    url = "https://gdpr-info.eu/art-17-gdpr/"  # Article 17: Right to erasure
    content = scrape_gdpr_article(url)

    if content:
        print("\n" + "=" * 50)
        print("Preview of scraped content:")
        print("=" * 50)
        # Show first 500 characters as a preview
        print(content[:500] + "..." if len(content) > 500 else content)
        print(f"\nTotal characters scraped: {len(content)}")
        print(f"Content saved to: gdpr_article_content.txt")

    # Demo 2: Batch scraping multiple articles (commented out by default)
    # Uncomment these lines to scrape Articles 22, 23, and 24
    # This demonstrates the URL pattern-based batch scraping feature
    # that's unique to this specialized scraper
    #
    # articles_to_scrape = [22, 23, 24]  # Article numbers to scrape
    # base_pattern = "https://gdpr-info.eu/art-{}-gdpr/"  # URL template
    # multiple_content = scrape_multiple_gdpr_articles(base_pattern, articles_to_scrape)
    # 
    # After running, each article would be in the returned dict:
    # multiple_content[22] -> Article 22 content
    # multiple_content[23] -> Article 23 content
    # multiple_content[24] -> Article 24 content