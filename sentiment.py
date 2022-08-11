import pandas as pd
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

tickers = ["TQQQ"]

def finviz_news(tickers):
    news_dict = {}
    news = []
    analyzer = SentimentIntensityAnalyzer()
    for ticker in tickers:
        finviz_url = f"https://finviz.com/quote.ashx?t={ticker}"
        finviz_req = Request(url=finviz_url, headers={'user-agent':'my-app'})
        finviz_response = urlopen(finviz_req)
        finviz_html = BeautifulSoup(finviz_response, 'html.parser')
        news_table = finviz_html.find(id="news-table")
        news_dict[ticker] = news_table
        news_rows = news_dict[ticker].findAll('tr')
        for i, row in enumerate(news_rows):
            news_title = row.a.text
            date_time = row.td.text.split(' ')
            if len(date_time) == 2:
                date = date_time[0]
                time = date_time[1]
            else:
                time = date_time[0]
            news.append([ticker, date, time, news_title])
        news_df = pd.DataFrame(data=news, columns=["Ticker", "Date","Time","News Title"])
        news_df["Sentiment Score"] = news_df['News Title'].apply(lambda title: analyzer.polarity_scores(title)['compound'])
        news_df['Date'] = pd.to_datetime(news_df['Date']).dt.date
    score_df = news_df.groupby(["Ticker", "Date"]).mean()
    print(score_df)
    score_df = score_df.unstack().xs('Sentiment Score', axis='columns').transpose()
    plt.rcParams["figure.figsize"] = (8,8)
    news_plot = score_df.tail(1).plot(kind="bar", width=0.5, align="center") 
    plt.style.use('fivethirtyeight')
    plt.title("News Sentiment Compound Score", fontsize=18)
    plt.xlabel("Date")
    plt.ylabel("Mean Sentiment Score")
    min_score, max_score = plt.ylim()
    plt.ylim(min_score,max_score+0.003)
    plt.axhline(y=0, color="k", linestyle="-",linewidth=0.5)
    plt.xticks(rotation=0)
    plt.legend().remove()
    for i, (bar,ticker) in enumerate(zip(news_plot.patches,tickers)):
        if bar.get_height() > 0:
            news_plot.annotate(format(bar.get_height(), ".3f"), 
                            (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va="center",
                            size=8, xytext=(0,4), textcoords = "offset points"
                            )
            news_plot.annotate(str(score_df.columns[score_df.iloc[-1]==bar.get_height()].values[0]), 
                            (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va="center",
                            size=8, xytext=(0,13), textcoords = "offset points"
                            )
        else:
            news_plot.annotate(format(bar.get_height(), ".3f"), 
                            (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va="center",
                            size=8, xytext=(0,-14), textcoords = "offset points"
                            )
            news_plot.annotate(str(score_df.columns[score_df.iloc[-1]==bar.get_height()].values[0]), 
                            (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va="center",
                            size=8, xytext=(0,-6), textcoords = "offset points"
                            )
    plt.show()

finviz_news(tickers)