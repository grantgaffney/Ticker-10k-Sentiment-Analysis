import React, { useState, useEffect } from "react";
import axios from "axios";
import Plot from "react-plotly.js";

const App = () => {
  const [ticker, setTicker] = useState("");
  const [analyzedTicker, setAnalyzedTicker] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stockScroller, setStockScroller] = useState([]);

  useEffect(() => {
    const fetchStockScroller = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5000/api/stocks");
        setStockScroller(response.data);
      } catch (err) {
        console.error("Error fetching stock scroller data", err);
      }
    };

    fetchStockScroller();
  }, []);

  const handleFetchData = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const response = await axios.post("http://127.0.0.1:5000/api/data", {
        ticker: ticker,
      });
      setData(response.data);
      setAnalyzedTicker(ticker.toUpperCase());
    } catch (err) {
      setError(err.response?.data?.error || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-100">
      {/* Stock Scroller */}
      <div className="w-full bg-blue-500 text-white p-2 shadow-md">
        <marquee behavior="scroll" direction="left">
          {stockScroller.length > 0 ? (
            stockScroller.map((stock, index) => (
              <span key={index} className="mr-8">
                {stock.symbol}: ${stock.price}
              </span>
            ))
          ) : (
            <span>Loading stocks...</span>
          )}
        </marquee>
      </div>

      {/* Search Section */}
      <div className="flex flex-col items-center justify-center p-4 bg-gray-200">
        <div className="w-full max-w-4xl p-8 bg-white shadow-md rounded-md">
          <h1 className="text-2xl font-bold mb-4 text-center">
            MD&A Sentiment and Stock Performance Analysis
          </h1>
          <form onSubmit={handleFetchData} className="flex flex-col mb-8">
            <input
              type="text"
              className="border border-gray-300 rounded-md p-2 mb-4"
              placeholder="Enter a stock ticker (e.g., AAPL)"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              required
            />
            <button
              type="submit"
              className="bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600"
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </form>
          {error && <div className="text-red-500 mb-4">{error}</div>}
        </div>
      </div>

      {/* Results Section */}
      {data && (
        <div className="flex flex-col items-center justify-center p-4 bg-gray-200">
          <div className="w-full max-w-4xl p-8 bg-white shadow-md rounded-md">
            <h2 className="text-2xl font-bold mb-4 text-center">{`Results for ${analyzedTicker}`}</h2>
            <div className="bg-gray-100 p-4 rounded-md mb-8">
              <p>
                <strong>Average Displacement:</strong> {data.avg_displacement.toFixed(2)}
              </p>
              <p>
                <strong>Correlation:</strong> {data.correlation.toFixed(2)}
              </p>
            </div>
            <Plot
              data={[
                {
                  x: data.years,
                  y: data.sentiment_scores,
                  mode: "lines+markers",
                  name: "Sentiment Score",
                },
                {
                  x: data.years,
                  y: data.stock_changes,
                  mode: "lines+markers",
                  name: "Stock % Change",
                },
              ]}
              layout={{
                title: "MD&A Sentiment and Stock Price Change Over Time",
                xaxis: { title: "Year" },
                yaxis: { title: "Score / % Change" },
                legend: { title: { text: "Metrics" } },
              }}
              style={{ width: "100%", height: "100%" }}
            />
            <p className="text-gray-600 text-sm mb-8">
              * By comparing the sentiment from the MD&A section from one year with the stock performance of the next year, this tool could help determine when to buy stock. The MD&A section often provides insights into internal company performance and speculation, so a high correlation could indicate that the company is candid in its MD&A, potentially aiding impact investing.
            </p>
            <h3 className="text-lg font-bold mb-4 text-center mt-8">Analysis</h3>
            <div className="bg-gray-100 p-4 rounded-md mb-8">
              <p>{data.analysis}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
