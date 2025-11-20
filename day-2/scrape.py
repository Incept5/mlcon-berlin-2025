import requests
from bs4 import BeautifulSoup
import re


def scrape_website(url):
    """
    Generic website scraper that works with any URL

    Args:
        url: The URL to scrape

    Returns:
        Extracted text content or None if failed
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try common content selectors in order of specificity
        content_selectors = [
            'div[data-elementor-type="single-post"]',  # Elementor posts
            '.entry-content',  # WordPress content
            '.post-content',
            'main article',    # Semantic HTML
            'main',
            'article',
            '.content',
            '.main-content',
            '#content',
            'body'  # Fallback
        ]

        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find('body')

        if not main_content:
            return None

        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""

        # Extract main heading
        main_heading = soup.find(['h1', 'h2'])
        heading_text = main_heading.get_text().strip() if main_heading else ""

        # Get all content elements
        content_elements = main_content.find_all([
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div', 'span'
        ])

        # Filter and process content
        text_parts = []
        processed_texts = set()

        # Add title if different from heading
        if title_text and title_text != heading_text:
            text_parts.append(f"# {title_text}\n")

        for element in content_elements:
            # Skip navigation, footer, header elements
            if element.find_parent(['nav', 'footer', 'header']):
                continue

            # Skip elements with navigation classes
            classes = ' '.join(element.get('class', [])).lower()
            if any(skip in classes for skip in ['nav', 'menu', 'footer', 'sidebar', 'widget']):
                continue

            text = element.get_text().strip()
            if not text or len(text) < 10:
                continue

            # Avoid duplicates
            normalized = ' '.join(text.split())
            if normalized in processed_texts:
                continue
            processed_texts.add(normalized)

            # Format based on element type
            if element.name.startswith('h'):
                level = int(element.name[1]) if element.name[1].isdigit() else 2
                prefix = '#' * min(level, 6)
                text_parts.append(f"\n{prefix} {text}\n")
            elif element.name == 'li':
                text_parts.append(f"- {text}")
            else:
                text_parts.append(text)

        # Join and clean up
        final_text = '\n'.join(text_parts)
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)
        final_text = re.sub(r' {2,}', ' ', final_text)

        return final_text.strip()

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def scrape_multiple_websites(urls):
    """
    Scrape multiple websites using the generic scraper

    Args:
        urls: List of URLs to scrape, or dict with labels as keys

    Returns:
        Dict with results
    """
    results = {}

    # Handle both list and dict inputs
    if isinstance(urls, list):
        url_dict = {f"Site {i+1}": url for i, url in enumerate(urls)}
    else:
        url_dict = urls

    for label, url in url_dict.items():
        print(f"\nScraping {label}...")

        content = scrape_website(url)
        if content:
            results[label] = content
            print(f"✓ Successfully scraped {label} ({len(content)} characters)")
        else:
            print(f"✗ Failed to scrape {label}")

    return results


if __name__ == "__main__":
    # Single website scraping demo
    print("=== SINGLE WEBSITE DEMO ===")
    url = "https://gdpr-info.eu/art-17-gdpr/"
    content = scrape_website(url)

    if content:
        print("\n" + "=" * 50)
        print("Preview of scraped content:")
        print("=" * 50)
        print(content[:500] + "..." if len(content) > 500 else content)
        print(f"\nTotal characters scraped: {len(content)}")

    # Multiple websites scraping demo with diverse sites
    print("\n\n=== MULTIPLE WEBSITES DEMO ===")
    websites_to_scrape = [
        "https://gdpr-info.eu/art-17-gdpr/",
        "https://gdpr-info.eu/art-18-gdpr/",
        "https://gdpr-info.eu/art-19-gdpr/"
    ]

    multiple_content = scrape_multiple_websites(websites_to_scrape)

    # Show detailed summaries
    if multiple_content:
        print("\n" + "=" * 60)
        print("DETAILED SUMMARIES")
        print("=" * 60)
        for label, content in multiple_content.items():
            print(f"\n{'-' * 40}")
            print(f"📄 {label}")
            print(f"{'-' * 40}")
            if content:
                lines = content.split('\n')
                # Find the first substantial line (title)
                title = next((line.strip('# ') for line in lines if line.strip() and len(line.strip()) > 5), "No title")

                print(f"Title: {title}")
                print(f"Length: {len(content)} characters")
                print(f"Preview:")
                print(content[:400] + "..." if len(content) > 400 else content)
            else:
                print("❌ No content available")
    else:
        print("No websites were successfully scraped.")