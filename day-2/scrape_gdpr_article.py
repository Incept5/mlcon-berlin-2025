import requests
from bs4 import BeautifulSoup
import re


def scrape_gdpr_article(url):
    """
    Scrape GDPR article content from gdpr-info.eu
    """

    # Add a user agent to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Send a GET request to the URL
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try multiple selectors to find the main content
        content_selectors = [
            'div[data-elementor-type="single-post"]',  # Original selector
            '.entry-content',  # Common WordPress content class
            '.post-content',  # Alternative content class
            'main',  # Main content element
            'article',  # Article element
            '.content',  # Generic content class
            'body'  # Fallback to body
        ]

        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                print(f"Found content using selector: {selector}")
                break

        if not main_content:
            print("Could not find main content container, using entire body")
            main_content = soup.find('body')

        if main_content:
            text_content = []

            # Extract the article title if present
            title = soup.find('title')
            if title:
                title_text = title.get_text().strip()
                text_content.append(f"# {title_text}\n")

            # Look for article headings specifically
            article_heading = soup.find(['h1', 'h2'], string=re.compile(r'Article \d+', re.IGNORECASE))
            if article_heading:
                text_content.append(f"## {article_heading.get_text().strip()}\n")

            # Find all relevant content elements in order
            content_elements = main_content.find_all([
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'p', 'li', 'div', 'span', 'td', 'th'
            ])

            # Filter out navigation, footer, and other non-content elements
            filtered_elements = []
            for element in content_elements:
                # Skip elements that are likely navigation or metadata
                if element.find_parent(['nav', 'footer', 'header']):
                    continue

                # Skip elements with certain classes that indicate non-content
                element_classes = element.get('class', [])
                skip_classes = ['nav', 'menu', 'footer', 'sidebar', 'widget', 'meta']
                if any(skip_class in ' '.join(element_classes).lower() for skip_class in skip_classes):
                    continue

                # Get text content
                text = element.get_text().strip()
                if not text or len(text) < 3:  # Skip very short or empty elements
                    continue

                # Skip if text is just a single character or number
                if len(text) == 1:
                    continue

                filtered_elements.append((element, text))

            # Process filtered elements
            processed_texts = set()  # To avoid duplicates
            for element, text in filtered_elements:

                # Skip if we've already processed this exact text
                if text in processed_texts:
                    continue
                processed_texts.add(text)

                # Format based on element type
                if element.name and element.name.startswith('h'):
                    # Determine heading level
                    level = int(element.name[1]) if element.name[1].isdigit() else 2
                    heading_prefix = '#' * min(level, 6)
                    text_content.append(f"\n{heading_prefix} {text}\n")
                elif element.name == 'li':
                    text_content.append(f"- {text}")
                elif element.name in ['td', 'th']:
                    # Handle table cells
                    if element.parent and element.parent.name == 'tr':
                        continue  # We'll handle tables separately
                else:
                    # Regular paragraph or div content
                    text_content.append(text)

            # Handle tables separately to maintain structure
            tables = main_content.find_all('table')
            for table in tables:
                text_content.append("\n### Table\n")
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_text = ' | '.join(cell.get_text().strip() for cell in cells)
                        if row_text.strip():
                            text_content.append(f"| {row_text} |")
                text_content.append("")

            # Clean up and join the content
            final_text = '\n'.join(text_content)

            # Clean up excessive whitespace
            final_text = re.sub(r'\n{4,}', '\n\n\n', final_text)
            final_text = re.sub(r' {2,}', ' ', final_text)
            final_text = re.sub(r'^\s+', '', final_text, flags=re.MULTILINE)
            final_text = final_text.strip()

            # Save to file
            filename = "gdpr_article_content.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(final_text)

            print(f"Successfully scraped content and saved to {filename}")
            return final_text

        else:
            print("Could not find any content on the page.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error occurred during scraping: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def scrape_multiple_gdpr_articles(base_url_pattern, article_numbers):
    """
    Scrape multiple GDPR articles

    Args:
        base_url_pattern: URL pattern with {} placeholder for article number
        article_numbers: List of article numbers to scrape

    Returns:
        Dict with article numbers as keys and content as values
    """
    all_content = {}

    for article_num in article_numbers:
        url = base_url_pattern.format(article_num)
        print(f"\nScraping Article {article_num} from {url}")

        content = scrape_gdpr_article(url)
        if content:
            all_content[article_num] = content
            print(f"Successfully scraped Article {article_num}")
        else:
            print(f"Failed to scrape Article {article_num}")

    return all_content


if __name__ == "__main__":
    # Single article scraping
    url = "https://gdpr-info.eu/art-17-gdpr/"
    content = scrape_gdpr_article(url)

    if content:
        print("\n" + "=" * 50)
        print("Preview of scraped content:")
        print("=" * 50)
        print(content[:500] + "..." if len(content) > 500 else content)
        print(f"\nTotal characters scraped: {len(content)}")

    # Example: Scrape multiple articles
    # articles_to_scrape = [22, 23, 24]  # Article numbers
    # base_pattern = "https://gdpr-info.eu/art-{}-gdpr/"
    # multiple_content = scrape_multiple_gdpr_articles(base_pattern, articles_to_scrape)