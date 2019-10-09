###############################################################################
### regression imports                                                      ###
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import viz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import env


###############################################################################
### personal imports                                                        ###
###############################################################################

from env import host, user, password
from scipy import stats
import datetime as dt

###############################################################################
##################################  fancify  ##################################
###############################################################################

from colorama import init, Fore, Back, Style


txt=''
title_fancy = Style.BRIGHT + Fore.YELLOW + Back.BLUE
rule_fancy = Style.NORMAL + Fore.YELLOW + Back.BLUE
intro_fancy = Style.NORMAL + Fore.YELLOW + Back.BLACK
demo_fancy = Style.NORMAL + Fore.CYAN + Back.BLACK 
header_fancy = Style.BRIGHT + Fore.CYAN + Back.BLACK
code_fancy = Style.NORMAL + Fore.BLACK + Back.WHITE

def fancify(text, fancy=Style.BRIGHT):
    return f'{fancy}{str(text)}{Style.RESET_ALL}'


def print_title(text, fancy=title_fancy, upline=True, downline=True):
    if upline:
        print ()
    print(star_line(fancy))
    print(star_title(text, fancy))
    print(star_line(fancy))
    if downline:
        print()

def print_rule(text, fancy=rule_fancy, upline=True, downline=True):
    text_lines = text.split('\n')
    if upline:
        print()
    for line in text_lines:
        print(fancify(f'{line:<80s}', fancy))
    if downline:
        print()


def star_line(fancy=Style.BRIGHT):
    return(f'{fancy}{txt:*^80s}{Style.RESET_ALL}')


def star_title(text, fancy=Style.BRIGHT):
    title_text = f'  {text}  ' 
    star_text = f'{title_text:*^80s}' if len(text) < 70 else text
    return(f'{fancy}{star_text}{Style.RESET_ALL}')


def model_test(description, model, test, screen=True):
    if len(description) > 50:
        text_desc = fancify(f'{description}\n{txt:50s}',Fore.WHITE)
    else:
        text_desc = fancify(f'{description:50s}',Fore.WHITE)
    model_desc = fancify(f'{model:>14s}',demo_fancy)
    actual_desc = fancify(f'{test:>14s}',header_fancy)
    output = f'{text_desc} {model_desc} {actual_desc}'
    if screen:
        print(output)
    else:
        return output

def model_title(description):
    m = 'Model'
    t = 'Test'
    output = fancify(model_test(description=description,model=f'{m:-^14}',test=f'{t:-^14}',screen=False))
    print(output)


def make_pct(num):
    return f'{100*num:>.4}%'


def make_int(num):
    return f'{int(round(num)):>d}'


def clear_screen():
    print('\033[2J')


def frame_splain(dataframe, use_name='sample', sample_limit=10):
    print(f'{header_fancy}{use_name.upper()} DATA{Style.RESET_ALL}')
    if sample_limit and len(dataframe) > sample_limit:
        print(dataframe.sample(10))
    else:
        print(dataframe)
    print()
    print(f'{header_fancy}{use_name} details{str(dataframe.shape)}{Style.RESET_ALL}')
    print(dataframe.dtypes)

###############################################################################
### get db url                                                              ###
###############################################################################


def get_db_url(user, password, host, database):
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'

employees_url = get_db_url(user=user, password=password, host=host, database=get_database)

###############################################################################
###############################################################################
###############################################################################

clear_screen()

###############################################################################
###############################################################################
###############################################################################

print_title('Exercises')


print_rule('''As a customer analyst, I want to know who has spent the most money with us 
over their lifetime. I have monthly charges and tenure, so I think I will be 
able to use those two attributes as features to estimate total_charges. I 
need to do this within an average of $5.00 per customer.''')

###############################################################################

print_title('wrangle.py')

print_rule('''The first step will be to acquire and prep the data. Do your work for this 
exercise in a file named wrangle.py.''')

###############################################################################
###############################################################################
###############################################################################


print_rule('''1. Acquire customer_id, monthly_charges, tenure, and total_charges from 
telco_churn database for all customers with a 2 year contract.''')



###############################################################################
###############################################################################
###############################################################################


print_rule('''2. Walk through the steps above using your new dataframe. You may handle the 
missing values however you feel is appropriate.''')


###############################################################################
###############################################################################
###############################################################################


print_rule('''3. End with a python file wrangle.py that contains the function, 
wrangle_telco(), that will acquire the data and return a dataframe cleaned 
with no missing values.''')


