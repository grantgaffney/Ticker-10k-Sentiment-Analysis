Hello, my name is Grant and this is my project application for the vRA position at the Financial Services Innovation Lab! I hope you enjoy :)

TO DEMO VIEW THE DEPLOYMENT LINK ON THE TOP RIGHT OF THE REPO or at https://ticker-10k-sentiment-analysis.vercel.app

File Overview:

  data/ - This folder houses cached information about previous API calls to prevent multiple API calls for the same ticker. It also reduces runtime.
  sec-edgar-filings/ - This is the folder where 10-K forms are temporarily downloaded to while they are being processed before they are deleted programatically to reduce wasted space.
  app.py/ - This file is the crux of the project. This is where all the backend computation is done
  frontend/src/app.js/ - This file does all of the frontend work.

Project Overview:

  The impetus behind this project is to be able to programatically utilize an LLM to determine the coorelation between the sentiment of the MD&A section of the 10-k filing for a
  company and the percent change of the company's stock price over the following year. Not only could a high degree of coorelation prove the sentiment analysis of a 10-k to be a tool
  for advising investment in a company but it could also give key insights into how a company's management speaks inside their MD&A section which could indicate systematic
  over-promising or under-promising, etc. These are invaluable insights about a company.

Project Workflow Overview:

  So what happens when you click analyze on the front-end?

  1. The company's 10-K forms are downloaded
  2. The forms are sequentially processed
     2a. First the text for a given 10-K is cleaned (all html removed, extra white-space, odd characters, etc.)
     2b. Second the actual MD&A section is extracted using regex
     2c. Third the section is passed to the LLM for sentiment analysis grading
     2d. The results are stored in a dictionary where they are indexed by the year {{year} : {sentiment_score, stock_percent_change}}
  3. The 10-k files are deleted after they have all been processed
  4. The resulting dictionary of all of the years and each of their sentiment scores and stock percent changes are passed to the LLM for aggregate analysis. It looks for trends and
     analyzes the degree of coorelation and any systemic over or under promising and other key insights from the perspective of an investor
  5. The results are presented to the user.
      5a. The results are cached so this ticker doesn't need to be reanalyzed
  
Tech-Stack Justification:

  The tech stack powering this project is a combination of Python for the backend scripting, beautifulsoup for parsing, OpenAI for LLM API, a React front-end and Vercel for hosting.

  I chose to use python because of the easy libraries and the nature of the project is scripting. It seemed like the natural and easy choice. I choose to use OpenAI 
  for the LLM API because of the strength of their model with textual analysis and ease of use (i also had free credits :) ). Then I used React for the front-end
  because of its simplicity and my prior experience with it. React is very easy and creates quick and powerful web apps which fit the necessitations of the project.

Other:

  The project is fully functional. To use it, you must input your own OPENAI_API_KEY inside of app.py. After you have done that the app will be functional to interact
  with new companies. A group of companies (APPL, GOOG, V, PLTR, TSLA, etc.) are all cached so their insights will appear without inputting a key.
  



    
