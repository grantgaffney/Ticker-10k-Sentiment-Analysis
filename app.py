from bs4 import BeautifulSoup
import re
from openai import OpenAI
import yfinance as yf
import os
import tiktoken
from sec_edgar_downloader import Downloader
from flask import Flask, request, jsonify
import json
import numpy as np
from scipy.stats import pearsonr
from flask_cors import CORS
import shutil

def download(ticker):
    dl = Downloader("Company", "email@domain.com")

    dl.get("10-K", ticker, after="1995-01-01", before="2023-12-31", download_details=True)


client = OpenAI(api_key='API_KEY') #Put your key here


def extract_text_from_html(file_path):
    """
    Extracts and cleans text from an HTML file, removing all HTML tags, styles, scripts,
    inline attributes, and encoded sections that are likely not human-readable text.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')

        # Remove script, style, meta, and noscript elements
        for element in soup(["script", "style", "meta", "noscript"]):
            element.decompose()

        # Extract text and remove any remaining HTML tags
        text = soup.get_text(separator=' ', strip=True)

        # Regex to remove Excel-like data
        text = re.sub(r'EXCEL.*?END', '', text, flags=re.DOTALL)

        # Remove any residual inline styles or attributes (common in malformed HTML or from copy-pasting from word processors)
        text = re.sub(r'\{[\s\S]*?\}', '',
                      text)  # Removes anything that appears within curly braces (common in inline styles)
        text = re.sub(r'\[[^\]]*\]', '', text)  # Removes anything that appears within square brackets
        text = re.sub(r'\([^)]*\)', '', text)  # Removes anything in parentheses (often used in documents)
        text = re.sub(r'\<[^>]*\>', '', text)  # Extra check to strip anything that looks like a tag
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

        # Regex to remove encoded sections, adjust as necessary based on actual data observation
        text = re.sub(r'begin \d+ .+?\n[\s\S]+?\nend', '', text, flags=re.MULTILINE)

    return text


def clean_text(text):
    """
    Cleans the extracted text by removing special characters and multiple spaces.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_mdna(text):
    # Define regex patterns for the start and end of the MD&A section
    start_pattern = re.compile(
        r"Management s Discussion and Analysis of Financial Condition and Results of Operations", re.IGNORECASE)
    end_pattern = re.compile(r"Quantitative and Qualitative Disclosures About Market Risk", re.IGNORECASE)

    # Find all occurrences of the start and end patterns
    start_matches = list(start_pattern.finditer(text))
    end_matches = list(end_pattern.finditer(text))

    if start_matches and end_matches:
        # Get the last start match
        last_start_match = start_matches[-1]
        # Find the first end match that occurs after the last start match
        for end_match in end_matches:
            if end_match.start() > last_start_match.end():
                mdna_text = text[last_start_match.end():end_match.start()]
                return mdna_text.strip()

    return "MD&A section not found or markers do not match."


def process_file(file):
    html_text = extract_text_from_html(file)
    clean_html_text = clean_text(html_text)

    return clean_html_text


def extract_fiscal_year(cleaned_text):
    """
    Extracts the fiscal year from cleaned text using regex patterns.

    Args:
        cleaned_text (str): The cleaned text from which to extract the fiscal year.

    Returns:
        str: The extracted fiscal year, or an error message if not found.
    """
    # Use regex to find the fiscal year (case-insensitive)
    fiscal_year_pattern = re.compile(r"For the fiscal year ended ([a-zA-Z]+ \d{1,2} \d{4})", re.IGNORECASE)
    search_result = fiscal_year_pattern.search(cleaned_text)

    if search_result:
        # Extract the full date
        full_date = search_result.group(1)
        # Extract just the year part from the full date
        year = re.search(r"\d{4}", full_date).group()
        return year
    else:
        return "Fiscal year not found in the document."


def analyze_mdna_and_stock(year, percent_change_stock, mdna_text, results_dict):

    prompt = f"You should only return a single number and nothing else. Analyze the following text from the MD&A section for the year {year}, and provide a numeric sentiment score from -100 to 100 reflecting the company's performance where 100 is the most optimal performance and -100 is the worst performance:\n\n{mdna_text}"

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo"
    )

    try:
        # Attempt to extract a numeric sentiment score from the LLM output
        response_text = chat_completion.choices[0].message.content.strip()
        sentiment_score = int(re.search(r'(-?\d+)', response_text).group())
        sentiment_score = categorize_sentiment(sentiment_score)
    except (AttributeError, ValueError):
        # If unable to find or convert to an integer, handle the error
        sentiment_score = None
        print("Failed to extract a numeric sentiment score from the LLM response.")

    results_dict[year] = (sentiment_score, percent_change_stock)
    return results_dict


def categorize_sentiment(score):
    if score < -87.5:
        return -100
    elif score < -62.5:
        return -75
    elif score < -37.5:
        return -50
    elif score < -12.5:
        return -25
    elif score < 12.5:
        return 0
    elif score < 37.5:
        return 25
    elif score < 62.5:
        return 50
    elif score < 87.5:
        return 75
    else:
        return 100


def get_yearly_stock_performance(ticker, year):
    # Construct the date strings for the start and end of the year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Fetch the daily stock data for the specified year
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if not stock_data.empty:
        # Get the opening price of the first trading day and the closing price of the last trading day
        opening_price_first_day = stock_data.iloc[0]['Open']
        closing_price_last_day = stock_data.iloc[-1]['Close']

        # Calculate the percent change
        percent_change = ((closing_price_last_day - opening_price_first_day) / opening_price_first_day) * 100

        # Truncate the result and cap at 100
        truncated_percent_change = int(percent_change)
        capped_percent_change = min(truncated_percent_change, 100)

        return capped_percent_change
    else:
        return None


def process_files(root_dir, ticker):
    results_dict = {}

    ticker_path = os.path.join(root_dir, ticker, '10-K')
    if os.path.isdir(ticker_path):
        for year_folder in os.listdir(ticker_path):
            year_form_path = os.path.join(ticker_path, year_folder)
            full_submission_path = os.path.join(year_form_path, 'full-submission.txt')

            if os.path.isfile(full_submission_path):
                print("Processing", year_folder)
                # Process each full-submission.txt file
                cleaned_text = process_file(full_submission_path)


                # Extract the fiscal year from the document
                fiscal_year = extract_fiscal_year(cleaned_text)
                print(fiscal_year)

                if fiscal_year.isdigit():
                    # Get the stock performance for the given year
                    percent_change_stock = get_yearly_stock_performance(ticker, fiscal_year)

                    if percent_change_stock is not None:
                        # Analyze the MD&A section and update results dictionary
                        mdna_section = extract_mdna(cleaned_text)
                        truncated_mdna_section = truncate_string_to_max_tokens(mdna_section, "gpt-3.5-turbo")
                        results_dict = analyze_mdna_and_stock(fiscal_year, percent_change_stock,
                                                              truncated_mdna_section, results_dict)

        # After processing, delete the entire ticker_path folder
        try:
            path = os.path.join(root_dir, ticker)
            shutil.rmtree(path)
            print(f"Deleted folder: {path}")
        except Exception as e:
            print(f"Error deleting folder {path}: {e}")
    else:
        download(ticker)
        results_dict = process_files(root_dir, ticker)

    results_dict = sort_results_by_year(results_dict)
    return results_dict


def truncate_string_to_max_tokens(text, model="gpt-3.5-turbo", max_tokens=15000):
    """
    Truncate a single string to fit within the specified token limit for the given model.

    Args:
        text (str): Input text to be truncated.
        model (str): Model name, e.g., "gpt-3.5-turbo" or "gpt-4-turbo".
        max_tokens (int): Maximum token limit for the given model.

    Returns:
        str: Truncated text.
    """
    # Get encoding for the model
    encoding = tiktoken.encoding_for_model(model)

    # Encode the text and count tokens
    tokens = encoding.encode(text)

    # Truncate tokens if necessary
    truncated_tokens = tokens[:max_tokens]

    # Decode back to text
    truncated_text = encoding.decode(truncated_tokens)

    return truncated_text

def sort_results_by_year(results_dict):
    """
    Sorts the results dictionary by year.

    Args:
        results_dict (dict): The results dictionary to be sorted.

    Returns:
        dict: Sorted results dictionary by year.
    """
    # Convert to list of tuples and sort by year (integer conversion)
    sorted_tuples = sorted(results_dict.items(), key=lambda item: int(item[0]))

    # Convert back to a regular dictionary
    return dict(sorted_tuples)


# results = {'2008': (75, -56), '2006': (25, 17), '2019': (0, 88), '1999': (-50, 100), '2012': (0, 24), '2017': (50, 46), '2005': (50, 100), '2009': (50, 100), '2022': (75, -26), '2015': (75, -3), '2016': (-75, 12), '2007': (-25, 100), '2021': (50, 33), '2003': (-25, 48), '2002': (0, -36), '2018': (75, -8), '2020': (25, 80), '2011': (50, 24), '2023': (-25, 47), '1995': (0, -18), '2004': (50, 100), '2000': (0, -71), '2010': (75, 51), '1996': (0, -32), '1997': (0, -37), '2014': (25, 41), '2013': (75, 0)}
# results = sort_results_by_year(results)


# results = process_files("/Users/grantgaffney/PycharmProjects/TCKRAnalysis/sec-edgar-filings", "TSLA")
# print(results)


#{'1995': (0, -18), '1996': (0, -32), '1997': (-50, -37), '1999': (-50, 100), '2000': (-50, -71), '2002': (-50, -36), '2003': (-50, 48), '2004': (0, 100), '2005': (50, 100), '2006': (75, 17), '2007': (50, 100), '2008': (25, -56), '2009': (50, 100), '2010': (50, 51), '2011': (0, 24), '2012': (25, 24), '2013': (50, 0), '2014': (-25, 41), '2015': (75, -3), '2016': (-50, 12), '2017': (75, 46), '2018': (100, -8), '2019': (-50, 88), '2020': (25, 80), '2021': (75, 33), '2022': (75, -26), '2023': (-75, 47)}
#TSLA {'2010': (-75, 39), '2011': (-50, 6), '2012': (-75, 14), '2013': (75, 100), '2014': (50, 48), '2015': (-50, 6), '2016': (0, -7), '2017': (100, 44), '2018': (0, 7), '2019': (50, 35), '2020': (75, 100), '2021': (50, 48), '2022': (75, -67)}


# Load cached data or create an empty dictionary if not available
cache_file_path = os.path.join("data", "cached_data.json")
if os.path.exists(cache_file_path):
    with open(cache_file_path, 'r') as f:
        cached_data = json.load(f)
else:
    cached_data = {}

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

def fetch_and_analyze_ticker(ticker):
    if ticker in cached_data and "results" in cached_data[ticker] and "analysis" in cached_data[ticker]:
        return cached_data[ticker]

    root_dir = "sec-edgar-filings"
    results = process_files(root_dir, ticker)

    # Shift stock changes back by one year
    shifted_results = shift_stock_performance(results)

    sentiment_scores = [result[0] for result in shifted_results.values()]
    stock_changes = [result[1] for result in shifted_results.values()]
    avg_displacement = calculate_average_displacement(sentiment_scores, stock_changes)
    correlation = calculate_correlation(sentiment_scores, stock_changes)

    analysis = generate_analysis(shifted_results, correlation, avg_displacement)

    # Update cache
    cached_data[ticker] = {
        "results": shifted_results,
        "analysis": analysis
    }

    with open(cache_file_path, 'w') as f:
        json.dump(cached_data, f)

    return cached_data[ticker]


def calculate_average_displacement(sentiment_scores, stock_changes):
    differences = np.abs(np.array(sentiment_scores) - np.array(stock_changes))
    return np.mean(differences)

def calculate_correlation(sentiment_scores, stock_changes):
    if len(sentiment_scores) == 0 or len(stock_changes) == 0:
        return None
    correlation, _ = pearsonr(sentiment_scores, stock_changes)
    return correlation

def generate_analysis(results, correlation, avg_displacement):
    years = list(results.keys())
    sentiment_scores = [result[0] for result in results.values()]
    stock_changes = [result[1] for result in results.values()]

    prompt = f"""
    As an investor, analyze the following financial data from the Management's Discussion and Analysis (MD&A) section of the 10-K form. Provide a brief textual analysis considering the sentiment scores from the MD&A section and the stock price changes. Frame your response in terms of being an investor.

    Years: {years}
    Sentiment Scores (MD&A Analysis): {sentiment_scores}
    Stock Price Changes: {stock_changes}

    Average Displacement: {avg_displacement:.2f}
    Correlation: {correlation:.2f}
    Provide an extremely brief and clear insight. Also, indicate the degree to which future price jumps could be predicted by positive sentiment scores.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo"
    )
    return chat_completion.choices[0].message.content.strip()

def shift_stock_performance(results):
    """
    Shifts the stock performance back by one year to align with the sentiment scores.
    Excludes the last year as no next-year performance data is available.

    Args:
        results (dict): The results dictionary with years as keys.

    Returns:
        dict: Shifted results with sentiment scores aligned to the next year's stock changes.
    """
    sorted_years = sorted(results.keys())
    shifted_results = {sorted_years[i]: results[sorted_years[i + 1]] for i in range(len(sorted_years) - 1)}

    return shifted_results


@app.route('/api/data', methods=['POST'])
def get_data():
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    result = fetch_and_analyze_ticker(ticker.upper())
    if "results" not in result or "analysis" not in result:
        return jsonify({"error": f"No data found for ticker {ticker}"}), 404

    results = result["results"]
    analysis = result["analysis"]

    sentiment_scores = [result[0] for result in results.values()]
    stock_changes = [result[1] for result in results.values()]
    avg_displacement = calculate_average_displacement(sentiment_scores, stock_changes)
    correlation = calculate_correlation(sentiment_scores, stock_changes)

    years = list(results.keys())

    return jsonify({
        "years": years,
        "sentiment_scores": sentiment_scores,
        "stock_changes": stock_changes,
        "avg_displacement": avg_displacement,
        "correlation": correlation,
        "analysis": analysis
    })

@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NFLX", "META",
        "BRK-A", "V", "NVDA", "JPM", "JNJ", "WMT", "BAC", "DIS",
        "XOM", "KO", "PEP", "CSCO", "ORCL", "ADBE", "CRM", "PYPL",
        "NKE", "CMCSA", "PFE", "CVX", "ABT", "MCD", "ACN", "TMO",
        "TXN", "AVGO", "COST", "WFC", "IBM", "T", "HON", "QCOM"
    ]
    stocks = []
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if not data.empty:
            current_price = data['Close'][0]
            stocks.append({
                "symbol": symbol,
                "price": round(current_price, 2)
            })
    return jsonify(stocks)


if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    app.run(debug=True)