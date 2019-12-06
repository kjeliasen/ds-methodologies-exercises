###############################################################################
### local imports                                                           ###
###############################################################################

#from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
# import acquire as acq

def get_blog_articles():
    from acquire import get_soup
    return make_new_blog_request()

def make_dictionary_from_blog_article(
    url, 
    file_name,
    headers = {'User-Agent': 'Nothing suspicious'},
    cache=True,
    cache_age=False,
    soup_slurper='*'
):
    # make the request to the url variable
    # make a "soup" variable
    # isolate the title of the article, store it as a string called "title"
    # isolate the body text of the article (buy CSS selector), name the variable "body"
    
    #response = get(url, headers=headers)
    
    soup = get_soup(
        url = url,
        headers = headers,
        file_name = file_name,
        cache = cache,
        cache_age = cache_age,
        soup_slurper = soup_slurper
    )
    
    title = soup.title.get_text()
    body = soup.select('div.mk-single-content.clearfix')[0].get_text()
    
    return {
        "title": title,
        "body": body,
        "url": url
    }


# def get_blog_articles():
    


def make_new_blog_request(
    file_name_csv='./codeup_blog_posts.csv',
    file_name_bs='z_stash/codeup_blog_posts.html',
    headers = {'User-Agent': 'Nothing suspicious'},
    cache=True,
    cache_age=False
):
    urls = [
        "https://codeup.com/codeups-data-science-career-accelerator-is-here/",
        "https://codeup.com/data-science-myths/",
        "https://codeup.com/data-science-vs-data-analytics-whats-the-difference/",
        "https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/",
        "https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/",
    ]

    output = []
    
    for url in urls:
        output.append(
            make_dictionary_from_blog_article(
                url = url, 
                file_name = file_name_bs, 
                headers = headers, 
                cache = cache, 
                cache_age = cache_age,
                soup_slurper = soup_slurper
            ))
    
    df=pd.DataFrame(output)
    df.to_csv(file_name_csv)
    return df


if __name__ == '__main__':
    print('opened wrangle_codeup_blog')
else:
    print('reached wrangle_codeup_blog')