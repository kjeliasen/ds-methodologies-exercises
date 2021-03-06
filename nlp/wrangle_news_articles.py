###############################################################################
### local imports                                                           ###
###############################################################################

#from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
# import acquire as acq


default_news_urls = [
        "https://inshorts.com/en/read/business",
        "https://inshorts.com/en/read/sports",
        "https://inshorts.com/en/read/technology",
        "https://inshorts.com/en/read/entertainment"
    ]


def get_news_articles(
    url='https://inshorts.com/en/read/', 
    url_list=default_news_urls,
    file_name_csv='./articles.csv',
    file_name_bs='z_stash/articles.html',
    headers={'User-Agent': 'Nothing suspicious'},
    slurper='.news-card',
    cache=True,
    cache_age=False
):
    from acquire import get_soup
    return make_new_news_request(
        url_list = url_list,
        headers = headers,
        file_name_csv = file_name_csv,
        file_name_bs = file_name_bs,
        cache = cache,
        cache_age = cache_age,
        slurper = slurper
    )


def make_dictionary_from_news_article(
    url='https://inshorts.com/en/read/', 
    file_name='z_stash/articles.html',
    headers={'User-Agent': 'Nothing suspicious'},
    slurper='.news_card',
    cache=True,
    cache_age=False
):
    from acquire import get_soup
    # make the request to the url variable
    # make a "soup" variable
    # isolate the title of the article, store it as a string called "title"
    # isolate the body text of the article (buy CSS selector), name the variable "body"
    
    #response = get(url, headers=headers)
    output = []
    
    soup = get_soup(
        url = url,
        headers = headers,
        file_name = file_name,
        cache = cache,
        cache_age = cache_age,
        slurper = slurper
    )
    
    articles = soup.select(slurper)
    
    for article in articles:
        title = article.select("[itemprop='headline']")[0].get_text()
        content = article.select("[itemprop='articleBody']")[0].get_text()
        author = article.select(".author")[0].get_text()
        published_date = article.select(".time")[0]["content"]
        category = url.split("/")[-1]
        article_data = {
            'title': title,
            'content': content,
            'category': category,
            'author': author,
            'published_date': published_date,
            'url': url
        }
        output.append(article_data)
    
    return output


def make_new_news_request(
    url_list=default_news_urls,
    file_name_csv='./articles.csv',
    file_name_bs='z_stash/articles.html',
    headers={'User-Agent': 'Nothing suspicious'},
    slurper='.news-card',
    cache=True,
    cache_age=False
):
    import pandas as pd
    urls = url_list

    output = []
    
    for url in urls:
        output.extend(
            make_dictionary_from_news_article(
                url, 
                file_name_bs, 
                slurper=slurper,
                headers=headers, 
                cache=cache, 
                cache_age=cache_age,
            ))
    
    df=pd.DataFrame(output)
    df.to_csv(file_name_csv)
    return df


if __name__ == '__main__':
    print('opened wrangle_news_articles')
else:
    print('reached wrangle_news_articles')

