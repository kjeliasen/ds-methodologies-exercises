###############################################################################
### local imports                                                           ###
###############################################################################

#from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
# import acquire as acq

def get_news_articles():
    from acquire import get_soup
    return make_dictionary_from_news_article()


def make_dictionary_from_news_article(
    url='https://inshorts.com/en/read/', 
    file_name='z_stash/articles.html',
    headers={'User-Agent': 'Nothing suspicious'},
    soup_slurper='.news_card',
    cache=True,
    cache_age=False
):
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
        soup_slurper = soup_slurper
    )
    
    articles = soup.select(soup_slurper)
    
    for article in articles:
        title = article.select("[itemprop='headline']")[0].get_text()
#         print(title)
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


# data = get_soup(
#     url = 'https://inshorts.com/en/read/',
#     headers = {'User-Agent': 'Nothing suspicious'},
#     file_name = 'z_stash/article.html',
#     soup_slurper='.news-card',
#     cache=False,
#     cache_age=False,
#    )



def make_new_news_request(
    file_name_csv='./articles.csv',
    file_name_bs='z_stash/articles.html',
    headers={'User-Agent': 'Nothing suspicious'},
    soup_slurper='.news-card',
    cache=True,
    cache_age=False
):
    urls = [
        "https://inshorts.com/en/read/business",
        "https://inshorts.com/en/read/sports",
        "https://inshorts.com/en/read/technology",
        "https://inshorts.com/en/read/entertainment"
    ]

    output = []
    
    for url in urls:
        output.extend(
            make_dictionary_from_news_article(
                url, 
                file_name_bs, 
                soup_slurper=soup_slurper,
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

